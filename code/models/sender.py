"""
Speaker models: a GRU language model as in "Emergent Communication of
    Generalizations" (https://arxiv.org/abs/2106.02668), and Transformer
    language models in two arms.

`SenderTransformerLM` is one architecture behind one flag, `bidirectional`,
    which selects a mask. Both arms cross-attend the two prototypes into a
    latent array and run the same blocks over it; the message is the array's
    tail. `true` reads that tail in one shot, `false` generates it in order,
    overwriting each slot with its symbol as it commits. See
    docs/architecture.md.

The Gumbel channel -- `layer_norm_logits`, `logit_scale`, `uniform_weight`,
    `sampling_tau` and their diagnostics -- is documented in docs/channel.md.
"""

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

import data
import data.language

import broccoli

from . import model_util


def trim_messages(token_id_rows):
    """
    Turn rows of decoded token ids into ragged content-token sequences: drop the
    leading SOS and truncate at the first EOS. Accepts anything
    iterable-of-iterables; returns a list of python int lists.
    """
    sos, eos = data.language.SOS_IDX, data.language.EOS_IDX
    trimmed = []
    for row in token_id_rows:
        toks = []
        for t in row:
            t = int(t)
            if t == sos:
                continue
            if t == eos:
                break
            toks.append(t)
        trimmed.append(toks)
    return trimmed


# Well below `F.layer_norm`'s 1e-5 default, so the normaliser keeps normalising
#     as a speaker's logit scale collapses. See docs/channel.md.
LAYER_NORM_EPS = 1e-12

# The bisection in `logit_scale`, sized so the solve is deterministic and
#     resolves to better than one part in 10^4. See docs/channel.md.
ENERGY_SOLVE_SAMPLES = 65536
ENERGY_SOLVE_SEED = 0
ENERGY_SOLVE_STEPS = 48
ENERGY_SCALE_MIN = 1e-3
ENERGY_SCALE_MAX = 1e3


def layer_norm_logits(logits: torch.Tensor, vocabulary: int) -> torch.Tensor:
    """
    Normalise the *emittable* vocabulary logits to zero mean and unit variance,
        per example and per position.

    Only the last `vocabulary` columns are normalised; the four reserved slots
        are concatenated back untouched, so they cannot pollute the statistics
        of the tokens that can actually be emitted.

    Functional and affine-free, so the transform is argmax-preserving and adds
        nothing to the `state_dict`. The speaker's two learnable channel
        parameters sit on opposite sides of it, and `eps` is load-bearing --
        see docs/channel.md.

    Args:
        logits: (..., vocabulary + 4), reserved tokens first
        vocabulary: number of emittable tokens

    Returns:
        A tensor of the same shape, with the emittable slice normalised
    """
    return torch.cat(
        [
            logits[..., :4],
            F.layer_norm(logits[..., 4:], (vocabulary,), eps=LAYER_NORM_EPS),
        ],
        dim=-1,
    )


def initial_logit_sample(vocabulary: int) -> torch.Tensor:
    """
    A fixed draw of the logits a *freshly initialised* speaker produces, which
        after `layer_norm_logits` are i.i.d. standard normal.

    Drawn once from a fixed seed and reused across the whole bisection in
        `logit_scale`, so the solve is deterministic and exactly monotone in the
        scale. See docs/channel.md.

    Args:
        vocabulary: number of emittable tokens

    Returns:
        A (samples, vocabulary) tensor of standard normal logits
    """
    generator = torch.Generator().manual_seed(ENERGY_SOLVE_SEED)
    return torch.randn(
        ENERGY_SOLVE_SAMPLES, vocabulary, generator=generator
    )


def initial_energy(
    scale: float,
    vocabulary: int,
    uniform_weight: float,
    sample: Optional[torch.Tensor] = None,
) -> float:
    """
    The fraction of the maximum possible entropy a fresh speaker's per-position
        distribution retains, once scaled and uniform-mixed.

    `H(p) / log2(V)`, so 1.0 is a uniform speaker and 0.0 one that emits a
        single token with certainty. This is what `init_energy` names.

    Args:
        scale: the multiplier applied to the normalised logits
        vocabulary: number of emittable tokens
        uniform_weight: as in `flatten_logit_distribution`
        sample: reuse a draw from `initial_logit_sample` rather than taking a
            fresh one -- the bisection passes the same sample at every step

    Returns:
        The retained entropy fraction, in [0, 1]
    """
    if sample is None:
        sample = initial_logit_sample(vocabulary)

    probabilities = torch.softmax(scale * sample, dim=-1)
    probabilities = (
        (1.0 - uniform_weight) * probabilities + uniform_weight / vocabulary
    )
    entropy = -(
        probabilities * probabilities.clamp_min(1e-30).log2()
    ).sum(-1).mean()

    return (entropy / math.log2(vocabulary)).item()


def logit_scale(
    init_energy: float, vocabulary: int, uniform_weight: float
) -> float:
    """
    The constant the normalised logits are multiplied by before sampling,
        resolved by bisection so that a fresh speaker retains `init_energy` of
        its maximum entropy.

    This is the exploration control. Why entropy rather than capacity or a
        symbol error rate, why a numerical solve rather than a closed form, and
        how the default was arrived at are all in docs/channel.md.

    Args:
        init_energy: `sender_language_model.init_energy` from the config,
            a fraction in (0, 1] -- 0.9 means "retain 90% of maximum entropy"
        vocabulary: number of emittable tokens
        uniform_weight: as in `flatten_logit_distribution`

    Returns:
        The multiplier, a plain float. It is the speaker's *initial* scale: each
            speaker stores its log in `log_logit_scale` and learns from there.
    """
    sample = initial_logit_sample(vocabulary)

    floor = initial_energy(
        ENERGY_SCALE_MAX, vocabulary, uniform_weight, sample
    )
    if init_energy < floor:
        warnings.warn(
            f"`init_energy` of {init_energy} is below the floor of "
            f"{floor:.4f} imposed by `uniform_weight` of {uniform_weight} at a "
            f"vocabulary of {vocabulary}. The scale will pin at "
            f"{ENERGY_SCALE_MAX} and the speaker will start at the floor "
            f"rather than at the requested entropy. Lower `uniform_weight` to "
            f"ask for less."
        )

    low, high = ENERGY_SCALE_MIN, ENERGY_SCALE_MAX
    for _ in range(ENERGY_SOLVE_STEPS):
        middle = math.sqrt(low * high)
        if initial_energy(middle, vocabulary, uniform_weight, sample) > (
            init_energy
        ):
            low = middle
        else:
            high = middle

    return math.sqrt(low * high)


def mask_reserved_tokens(logits: torch.Tensor) -> torch.Tensor:
    """
    Set the four reserved tokens (PAD/SOS/EOS/UNK) to -inf so they can never be
        emitted mid-message. SOS and EOS are attached by the caller instead, so
        messages are fixed-length.

    Out of place, because this runs on the output of the vocabulary projection
        and writing -inf into that in place would modify a tensor autograd still
        needs. Runs before the exploration noise so the uniform mixture is
        spread over the emittable tokens only.

    Args:
        logits: (..., vocabulary + 4), reserved tokens first

    Returns:
        A tensor of the same shape with the reserved positions set to -inf
    """
    reserved = torch.zeros_like(logits, dtype=torch.bool)
    reserved[..., :4] = True
    return logits.masked_fill(reserved, -float("inf"))


def flatten_logit_distribution(
    logits: torch.Tensor,
    uniform_weight: float
) -> torch.Tensor:
    """
    Mix a uniform distribution into `logits` at weight `uniform_weight`, in log
        space.

    The uniform component is spread over the emittable tokens only, i.e. those
        not already masked to -inf; spreading it over all `vocabulary + 4` slots
        and masking afterwards would silently deliver less than the nominal
        weight. Masked positions stay masked.

    Returns:
        Logits whose absolute differences are reduced, i.e. a less certain
            distribution
    """
    emittable = torch.isfinite(logits)
    normalised_logits = F.log_softmax(logits, dim=-1)

    # Make a uniform distribution, but in the log space
    uniform_log_probs = -torch.log(
        emittable.sum(-1, keepdim=True).to(logits.dtype)
    ).expand_as(logits)

    # Mix the log distributions, like log( w * uniform + (1-w) * model ).
    # Masked entries are -inf in both components, and `logsumexp` of two -inf
    #     backpropagates NaN, so they are mixed as a finite placeholder and the
    #     mask restored afterwards. `torch.where` routes the gradient to the
    #     selected branch only, so the placeholder never reaches the speaker.
    placeholder = torch.zeros_like(logits)
    combined_logits = torch.stack(
        [
            torch.where(emittable, uniform_log_probs, placeholder)
            + np.log(uniform_weight),
            torch.where(emittable, normalised_logits, placeholder)
            + np.log(1 - uniform_weight),
        ],
        dim=-1,
    )

    return torch.where(
        emittable, torch.logsumexp(combined_logits, dim=-1), logits
    )


def mean_winning_probability(
    logits: torch.Tensor,
    scale: torch.Tensor,
    uniform_weight: float,
) -> torch.Tensor:
    """
    The fraction of symbols that survive the Gumbel noise, averaged over slots,
        i.e. the `realised_survival` column.

    By the Gumbel-max identity this is just the winning token's softmax
        probability, so no Monte Carlo is needed; `tests/test_exploration.py`
        pins the identity. Applies the real pipeline in the real order -- scale
        first, then the uniform mixture -- so the mixture's bounds hold.

    Purely a measurement; pass the scale detached.

    Args:
        logits: (..., vocabulary + 4), reserved tokens already masked to -inf
        scale: the multiplier applied before mixing, i.e. `logit_scale`,
            detached
        uniform_weight: as in `flatten_logit_distribution`

    Returns:
        A scalar tensor, the mean over all slots of the winning token's
            post-mixing probability
    """
    scaled = logits * scale
    if uniform_weight > 0.0:
        scaled = flatten_logit_distribution(scaled, uniform_weight)
    return scaled.softmax(-1).max(-1).values.mean()


class ExampleContrast(nn.Module):
    """
    Let the referents inform each other before they are pooled.

    Both prototypers work *within* a polarity -- `AveragePrototyper` means each
        half and `AttentionPrototyper` scores each half with its own
        `SequencePool` -- so nothing in the speaker compares a positive example
        against a negative one. The only place the two halves meet is the
        language model's cross-attention, by which point each is already a
        single vector. This stage runs one self-attention over all `2n`
        referents at once and adds the result back as a residual, so a positive
        example can be represented by what distinguishes it from the negatives
        rather than by what it is in isolation.

    The prototyper downstream is unchanged and still receives
        `(batch, 2 * n_positive, referent embedding size)`; either of them
        composes with this.

    **What it costs, and why the diagnostics below exist.** The message becomes
        a function of the sampled negatives rather than of the concept alone, so
        the same concept with different distractors gets a different message --
        which is exactly what `topsim` penalises. This stage can therefore raise
        accuracy and lower compositionality at the same time, and
        `contrast_share` and `contrast_within_share` are what make that a
        reportable result rather than an inference from accuracy.

    **It opens at exactly the identity.** `contrast_gate` starts at zero, so a
        run with this stage on is bit-identical to one without it at step 0 and
        the arm is an ablation of one thing. The gate is a plain scalar and
        deliberately *not* log-parameterised: `exp` cannot reach zero, and zero
        is the whole point. Its sign is free because the branch's direction is
        arbitrary -- a negative gate is the same branch pointing the other way.

    **Why a gate rather than a zero-initialised projection.** Both open at the
        identity, but a zero matrix does not travel. AdamW moves a parameter by
        about `lr` per step whatever the gradient's size, so `out_projection`
        would have to climb from 0 to its own init scale `1/sqrt(d_model)` one
        `lr`-step at a time: 560 steps of perfectly sign-consistent gradient at
        `lr` 1e-4, which on birds' 62 optimiser steps an epoch is nine epochs of
        flat, optimistically. That is the arithmetic that made the logit scale's
        traverse the bottleneck for those runs. A lone scalar at
        `contrast_gate_lr` (2e-3) reaches 0.1 in fifty steps instead, and
        `out_projection` starts at a properly scaled random direction, so the
        branch contributes at a sensible magnitude the moment the gate opens
        rather than having to build one first.

        A gate at zero is a starting point and not a weld:
        `dL/dgate = <branch, dL/dout>` is non-zero there. Compare
        `AttentionDiscriminator.mix_floor`, which is in the parameterisation for
        the same reason a `clamp` would not do.

    **Its own projection and norm**, as the listener's slots have and for the
        same reason. `bias=False` on the adapter is load-bearing: the norm can
        only divide the backbone's scale out exactly if what reaches it is
        homogeneous in the input, which is what makes the *rate* of departure
        comparable across backbones. The residual is over the raw referents, so
        what the prototyper pools is still at the backbone's own scale -- the
        same "score from normalised selves, weight over raw selves" split
        `AttentionPrototyper` makes.

    **Polarity reaches this stage through the tag and nowhere else.** With
        `rotary_embedding=None` the attention is permutation-equivariant, so it
        cannot read the first-half-positive ordering that the rest of the
        speaker relies on; `label_embedding` is the only route, and it is
        indexed from the labels rather than from the halving index.
        `tests/test_contrast.py` pins both halves of that.
    """

    def __init__(
        self,
        referent_embedding_size,
        **kwargs
    ):
        """
        Args:
            referent_embedding_size: width of the backbone's output, which is
                also the width this returns -- the residual fixes it.
        """
        super().__init__()
        self.referent_embedding_size = referent_embedding_size
        self.d_model = kwargs["d_model"]
        self.heads = kwargs["heads"]
        self.self_attention_dropout = kwargs["self_attention_dropout"]

        self.adapter = nn.Linear(
            self.referent_embedding_size,
            self.d_model,
            bias=False
        )
        self.layer_norm = nn.LayerNorm(
            self.d_model, elementwise_affine=False, eps=LAYER_NORM_EPS
        )

        # Row 0 positive, row 1 negative, as on `SenderTransformerLM`. The name
        #     must keep "embedding" in it or `gradboard` will start decaying it,
        #     and it is deliberately not `polarity_embedding`:
        #     `SPLIT_LEARNING_RATES` selects by suffix, so any name ending that
        #     way -- `contrast_polarity_embedding` included -- would silently
        #     join the speaker tag's parameter group. See docs/architecture.md.
        self.label_embedding = nn.Parameter(torch.zeros(2, self.d_model))

        # Every broccoli argument is set explicitly, including the inert ones.
        #     See docs/broccoli.md.
        self.attention = broccoli.transformer.MHAttention(
            self.d_model,
            self.heads,
            dropout=self.self_attention_dropout,
            causal=False, # Every referent may inform every other one
            seq_len=None, # Only read when causal; the set has no fixed size
            linear_module=nn.Linear,
            bos_tokens=0,
            knocking_heads=False,
            # No positional information, and load-bearing: referent order *is*
            #     the label vector here, so a stage able to index its own
            #     sequence axis could read polarity without the tag. `causal` is
            #     off for the same reason it is off in the listener's stacks.
            #     `positional_heads` is inert and pinned at the repo-wide 1.0.
            rotary_embedding=None,
            positional_heads=1.0,
            source_size=None,
            scaling="d",
        )

        self.out_projection = nn.Linear(self.d_model, self.referent_embedding_size)

        self.contrast_gate = nn.Parameter(torch.zeros(()))

        # Per-batch diagnostics, read by `train.py` for metrics.csv. See
        #     docs/measurement.md.
        self.contrast_share = float("nan")
        self.contrast_within_share = float("nan")

        self.reset_parameters()

    def forward(self, samples, labels):
        """
        Args:
            samples: (batch, n_examples, referent embedding size), the first
                half positive and the rest negative
            labels: (batch, n_examples), 1.0 positive and 0.0 negative. Read
                rather than assumed: `Sender.get_prototypes` has already
                checked that the two agree, so indexing the tag from the labels
                costs nothing and stays honest if that layout ever changes.

        Returns:
            A tensor of `samples`' shape, each referent informed by the rest of
                the set.
        """
        tag = self.label_embedding[(1.0 - labels).long()]
        adapted = self.layer_norm(self.adapter(samples)) + tag

        branch = self.out_projection(self.attention(adapted, adapted, adapted))
        contribution = self.contrast_gate * branch

        self._record_diagnostics(samples, branch, contribution)

        return samples + contribution

    @torch.no_grad()
    def _record_diagnostics(self, samples, branch, contribution):
        """
        Volume and shape, kept apart: `contrast_share` measures the gated
            contribution and `contrast_within_share` the branch before the gate,
            so a branch that is well-shaped but still quiet reads as exactly
            that rather than as noise. Undefined shares report 0.0 rather than
            NaN, since a NaN here would be indistinguishable from the stage
            being switched off.
        """
        referent_scale = samples.pow(2).mean().sqrt()
        self.contrast_share = (
            contribution.pow(2).mean().sqrt() / referent_scale
        ).item() if referent_scale > 0.0 else 0.0

        # The part of the branch that is example-level, i.e. what neither a
        #     vector common to the whole game nor a per-polarity offset could
        #     have produced. A common vector shifts both prototypes equally and
        #     the language model's `LayerNorm` eats most of it; a per-polarity
        #     offset is a learned "I am positive", which `AttentionPrototyper`'s
        #     two separate pools already provide. Only the remainder is contrast
        #     *between examples*, which is the whole point of the stage. The
        #     polarity means are nested inside the grand mean, so the two sums of
        #     squares are orthogonal and this is a share of the total.
        n_positive = branch.size(1) // 2
        positive, negative = branch[:, :n_positive], branch[:, n_positive:]
        polarity_means = torch.cat(
            (
                positive.mean(1, keepdim=True).expand_as(positive),
                negative.mean(1, keepdim=True).expand_as(negative),
            ),
            dim=1,
        )
        total = branch.pow(2).sum()
        within = (branch - polarity_means).pow(2).sum()
        self.contrast_within_share = (
            (within / total).item() if total > 0.0 else 0.0
        )

    def reset_parameters(self):
        self.adapter.reset_parameters()
        # A no-op while the norm is parameter-free, and listed anyway so that
        #     turning `elementwise_affine` back on cannot leave a reset speaker
        #     holding trained gains. See docs/anecdotes.md.
        self.layer_norm.reset_parameters()
        self.attention.reset_parameters()
        self.out_projection.reset_parameters()

        with torch.no_grad():
            # Antipodal at unit per-element variance, which is what the
            #     parameter-free norm above emits: the tag opens at the scale of
            #     what it is added to, with no constant to choose and none to
            #     keep in step with `d_model`. `SenderTransformerLM`'s
            #     `polarity_embedding` is initialised the same way and
            #     docs/architecture.md carries the argument.
            positive_tag = torch.randn(self.d_model)
            self.label_embedding.copy_(
                torch.stack([positive_tag, -positive_tag])
            )

            self.contrast_gate.zero_()


class AveragePrototyper(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        # Per-batch diagnostics, defined here as well as on
        #     `AttentionPrototyper` so both arms write the same columns.
        #     Averaging is pooling with uniform weights, so the effective count
        #     is the example count and there is no scoring vector to report.
        self.pool_effective_examples = float("nan")
        self.pool_score_norm = float("nan")

    def forward(self, samples, labels=None):
        """
        Args:
            samples: a tensor of shape (batch, n_examples, embedding size)
                where each example is an embedded image. n_examples must be
                even; the first half are positive and the rest negative.
            labels: unused -- the first half of provided examples is always
                positive. See `samples`.
        """
        n_pos_ex = samples.size(1) // 2

        positive_examples = samples[:, :n_pos_ex, :]
        negative_examples = samples[:, n_pos_ex:, :]

        positive_prototype = positive_examples.mean(1)
        negative_prototype = negative_examples.mean(1)

        self.pool_effective_examples = float(n_pos_ex)

        return positive_prototype, negative_prototype

    def reset_parameters(self):
        pass


class AttentionPrototyper(nn.Module):
    """
    Pool each polarity with `SequencePool`'s learned attention -- a single
        scoring direction per polarity, softmaxed over the examples -- rather
        than averaging them.

    Two departures from a bare `SequencePool`, both there to stop the softmax
        over examples inheriting a pre-softmax magnitude set by the backbone:
        zero-initialised scoring weights, so the rung opens at
        `AveragePrototyper`'s behaviour exactly; and a parameter-free
        `LayerNorm` on the scoring path only, so the *rate* of departure from
        the mean is comparable across arms. See docs/architecture.md.
    """

    def __init__(self, d_model, *args, **kwargs):
        super().__init__()
        self.d_model = d_model

        # Scoring path only. Shared by both polarities because it has no
        #     parameters to share.
        self.score_norm = nn.LayerNorm(d_model, elementwise_affine=False)

        self.pos_pool = broccoli.vit.SequencePool(d_model)
        self.neg_pool = broccoli.vit.SequencePool(d_model)

        # Per-batch diagnostics, read by `train.py` for metrics.csv. See
        #     docs/measurement.md.
        self.pool_effective_examples = float("nan")
        self.pool_score_norm = float("nan")

        self.reset_parameters()

    def _pool(self, pool, examples):
        """
        Attention-weighted sum of `examples`, scored from their normalised
            selves and weighted over their raw selves.

        Returns:
            (prototype, weights), the weights being needed for the diagnostic.
        """
        weights = pool.attention_scores(self.score_norm(examples))
        prototype = torch.einsum("bs,bsd->bd", weights, examples)
        return prototype, weights

    def forward(self, samples, labels=None):
        n_pos_ex = samples.size(1) // 2

        positive_examples = samples[:, :n_pos_ex, :]
        negative_examples = samples[:, n_pos_ex:, :]

        positive_prototype, positive_weights = self._pool(
            self.pos_pool, positive_examples
        )
        negative_prototype, negative_weights = self._pool(
            self.neg_pool, negative_examples
        )

        self._record_diagnostics(positive_weights, negative_weights)

        return positive_prototype, negative_prototype

    @torch.no_grad()
    def _record_diagnostics(self, positive_weights, negative_weights):
        effective = torch.cat(
            [
                1.0 / positive_weights.pow(2).sum(-1),
                1.0 / negative_weights.pow(2).sum(-1),
            ]
        ).mean()
        self.pool_effective_examples = effective.item()

        self.pool_score_norm = 0.5 * (
            self.pos_pool.attention[0].weight.norm().item()
            + self.neg_pool.attention[0].weight.norm().item()
        )

    def reset_parameters(self):
        self.pos_pool.reset_parameters()
        self.neg_pool.reset_parameters()

        # After broccoli's own reset, not instead of it, so any parameter
        #     `SequencePool` grows is still initialised the way broccoli
        #     intends. Only the scoring projection is overridden.
        with torch.no_grad():
            for pool in (self.pos_pool, self.neg_pool):
                pool.attention[0].weight.zero_()
                if pool.attention[0].bias is not None:
                    pool.attention[0].bias.zero_()


class GumbelChannel:
    """
    The exploration channel both speakers send through. See docs/channel.md.

    A mixin rather than a submodule: `log_logit_scale` stays registered on the
        speaker itself, so `state_dict` keys -- and every checkpoint written
        against them -- are unchanged.
    """

    def _init_channel(self, init_energy, vocabulary, uniform_weight):
        """
        Call from `__init__` where the parameter should be created: creation
            order fixes which RNG draw every later parameter gets, so moving
            this moves a speaker's initialisation.
        """
        # Stored as its log so that `exp` keeps it strictly positive.
        self.initial_logit_scale = logit_scale(
            init_energy, vocabulary, uniform_weight
        )
        self.log_logit_scale = nn.Parameter(
            torch.tensor(self.initial_logit_scale).log()
        )

        # Fraction of training elapsed, set once per epoch by `train.py`. A
        #     position in a schedule, not state: recovered from the epoch counter
        #     on resume.
        self.training_progress = 0.0

        self.reset_channel_diagnostics()

    def reset_channel_diagnostics(self):
        """
        Per-batch diagnostics for metrics.csv. On both speakers so the two write
            the same columns; `polarity_separation` is not-applicable rather
            than unused on the GRU. See docs/measurement.md.
        """
        self.realised_survival = float("nan")
        self.logit_spread = float("nan")
        self.polarity_separation = float("nan")

    def reset_logit_scale(self):
        """
        Put the learned scale back to the value `init_energy` solved for, so a
            reset does not leave a trained channel behind a fresh speaker.
        """
        with torch.no_grad():
            self.log_logit_scale.fill_(math.log(self.initial_logit_scale))

    @property
    def sampling_tau(self):
        """
        The temperature handed to `gumbel_softmax`: the configured `tau`,
            adjusted towards `tau * logit_scale / initial_logit_scale` by a
            cosine schedule over training.

            ratio  = max(logit_scale / initial_logit_scale, 1)
            weight = (1 + cos(pi * training_progress)) / 2
            tau    = configured_tau * (1 + weight * (ratio - 1))

        A run opens fully coupled and ends at exactly the configured `tau`. The
            schedule is open-loop on purpose, the ratio is floored at 1, and the
            scale is detached -- a differentiable tau would put inf into the
            gradient w.r.t. the scale. All of that, and what the coupling buys,
            is in docs/channel.md.
        """
        ratio = torch.clamp(
            self.logit_scale.detach() / self.initial_logit_scale, min=1.0
        )
        weight = 0.5 * (1.0 + math.cos(math.pi * min(self.training_progress, 1.0)))
        return self.tau * (1.0 + weight * (ratio - 1.0))

    @property
    def logit_scale(self):
        """
        The multiplier applied to the normalised logits before sampling, always
            positive. Read here rather than exponentiating at the use sites, so
            they and the survival diagnostic cannot drift apart.
        """
        return self.log_logit_scale.exp()

    def reserved_onehot(self, index, batch_size, device):
        onehot = torch.zeros(
            batch_size,
            1,
            self.vocabulary + 4, # +4 for PAD, SOS, EOS, UNK
            device=device
        )
        onehot[:, 0, index] = 1.0
        return onehot

    def logit_spread_of(self, logits):
        """Taken *before* normalisation, which is the whole point of it."""
        return logits[..., 4:].detach().float().std(-1).mean()

    def sample_symbols(self, logits):
        """
        Turn one step's (or one message's) logits into symbols.

        Returns `(onehot, pre_gain_logits)` -- the masked, normalised, *un*scaled
            logits the survival diagnostic is measured from, or None outside
            training. Callers pool it over positions themselves.

        The order is not interchangeable. Scaling the *masked* logits rather
            than re-masking after the scale sends `-inf` into the gradient
            w.r.t. the scale, which becomes NaN and makes `GradScaler` skip
            every step -- silently, except for a frozen `logit_spread`. See
            docs/channel.md.

        The gain goes on through `model_util.scale_without_attenuating`, so the
            forward is exactly `normalised * logit_scale` -- the fidelity of the
            channel against the fixed 1.283 Gumbel noise floor is the scale's
            real job and is untouched -- while the backward into `normalised`,
            and so into this speaker's whole stack and vision model, does not
            carry the scale as a factor. `dL/dlogit_scale` is unchanged, so the
            covariance `scripts/ignition_audit.py` measures still reads what it
            always did; what changes is that a scale sliding down no longer
            turns down the machinery that would have made raising it worthwhile.
            The listener's `ScoreVolume` is the same treatment on the same
            shape. See the helper's docstring.
        """
        normalised = layer_norm_logits(logits, self.vocabulary)
        masked = mask_reserved_tokens(normalised)

        if not self.training:
            # Greedy: eval measures the policy. The reserved tokens are -inf, so
            #     the argmax can never select one.
            return (
                F.one_hot(masked.argmax(-1), self.vocabulary + 4).to(masked.dtype),
                None,
            )

        scaled = mask_reserved_tokens(
            model_util.scale_without_attenuating(normalised, self.logit_scale)
        )

        if self.uniform_weight > 0.0:
            scaled = flatten_logit_distribution(scaled, self.uniform_weight)

        # `argmax(logits + noise)` with a straight-through gradient. `tau`
        #     rescales the *soft* sample only; the hard forward sample is an
        #     argmax and so invariant to it.
        onehot = F.gumbel_softmax(
            scaled,
            tau=self.sampling_tau,
            hard=True,
            dim=-1
        )

        return onehot, masked.detach()

    def record_survival(self, pre_gain_logits):
        self.realised_survival = mean_winning_probability(
            pre_gain_logits.float(),
            self.logit_scale.detach(),
            self.uniform_weight,
        ).item()


class SenderGRULM(GumbelChannel, nn.Module):
    def __init__(
        self,
        referent_embedding_size,
        **kwargs
    ):
        super().__init__()
        self.referent_embedding_size = referent_embedding_size
        self.token_embedding_size = kwargs["token_embedding_size"]
        self.d_model = kwargs["d_model"]
        self.vocabulary = kwargs["vocabulary"]
        self.message_length = kwargs["message_length"]
        # The configured tau. The tau actually passed to `gumbel_softmax` is
        #     the `sampling_tau` property below, which tracks the learned scale.
        self.tau = kwargs["tau"]
        self.uniform_weight = kwargs["uniform_weight"]
        self.dropout = kwargs["dropout"]
        self.layers = kwargs["layers"]
        self.bidirectional = kwargs["bidirectional"]
        self.directions = 2 if self.bidirectional else 1

        self._init_channel(
            kwargs["init_energy"], self.vocabulary, self.uniform_weight
        )

        self.gru = nn.GRU(
            self.token_embedding_size,
            self.d_model,
            num_layers=self.layers,
            bias=True,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )

        # The bias here is the speaker's token prior, and it is pre-norm on
        #     purpose -- see docs/channel.md.
        self.outputs2vocab = nn.Linear(
            self.d_model * self.directions,
            self.vocabulary + 4 # +4 for PAD, SOS, EOS, UNK
        )

        self.init_h = nn.Linear(
            2 * referent_embedding_size, 
            self.layers * self.directions * self.d_model
        )
        
        self.token_embedding = nn.Embedding(
            self.vocabulary + 4, # +4 for PAD, SOS, EOS, UNK
            self.token_embedding_size
        )

        self.reset_parameters()

    def decode(
        self,
        prototypes,
        return_states=True
    ):
        """
        Run the decoding loop once, returning both the message and the symbol
            embeddings that produced it.

        This speaker is autoregressive, so each embedding depends on the symbols
            sampled before it and the two only correspond when they come from
            the *same* call. `forward` discards the embeddings, so any analysis
            needing them paired must call this directly, as `Sender.speak` does.

        Returns:
            lang_tensor: (batch, message_length, vocabulary + 4)
            symbol_embeddings: (batch, message_length - 2, d_model * directions),
                one dense contextual vector per content symbol, taken before the
                vocabulary projection and sampling, or None when `return_states`
                is False.
        """
        batch_size = prototypes[0].size(0)
        device = prototypes[0].device

        # Initialize hidden state. Must be (num_layers * directions, B, H)
        concatenate_prototypes = torch.cat(prototypes, 1)
        states = (
            self.init_h(concatenate_prototypes)
                .view(batch_size, self.layers, self.directions, self.d_model)
                .permute(1, 2, 0, 3).contiguous() # (L, Dir, B, D)
                .view(self.layers * self.directions, batch_size, self.d_model)
        )

        lang = []
        symbol_embeddings = []
        # Pre-gain logits for every position, kept so the diagnostics are pooled
        #     once per batch rather than once per position.
        survival_logits = []
        spread_steps = []

        # Create and add SOS token
        sos_onehot = self.reserved_onehot(
            data.language.SOS_IDX, batch_size, device
        )  # Shape: (B, 1, V)
        lang.append(sos_onehot)

        gru_in = sos_onehot @ self.token_embedding.weight  # Shape: (B, 1, D)

        # Main sampling loop (fixed length of message_length - 2)
        for i in range(self.message_length - 2):

            gru_out, states = self.gru(gru_in, states)

            step_embedding = gru_out[:, -1, :] # Shape: (B, D * Dir)
            symbol_embeddings.append(step_embedding)

            logits = self.outputs2vocab(step_embedding) # Shape: (B, V)

            if self.training:
                spread_steps.append(self.logit_spread_of(logits))

            predicted_onehot, pre_gain = self.sample_symbols(logits)

            if self.training:
                survival_logits.append(pre_gain)

            # 6. Prepare next input
            lang.append(predicted_onehot.unsqueeze(1))
            gru_in = (predicted_onehot.unsqueeze(1)) @ self.token_embedding.weight # (B, 1, D)

        # Pooled over positions after the loop: per position instead would read
        #     each position's statistics alone.
        if self.training:
            self.record_survival(torch.stack(survival_logits, 1))
            self.logit_spread = torch.stack(spread_steps).mean().item()

        # Add final EOS token
        lang.append(
            self.reserved_onehot(data.language.EOS_IDX, batch_size, device)
        )

        # Concatenate
        lang_tensor = torch.cat(lang, 1)

        return (
            lang_tensor,
            torch.stack(symbol_embeddings, 1) if return_states else None
        )

    def forward(
        self,
        prototypes,
        **kwargs
    ):
        """
        No greedy or epsilon-greedy generation options: the former is only used
            by the ACRe parts of the original code, and the latter is off by
            default and not discussed in the original paper.
        """
        return self.decode(prototypes, return_states=False)[0]

    def reset_parameters(self):
        self.init_h.reset_parameters()
        self.gru.reset_parameters()
        self.outputs2vocab.reset_parameters()
        self.token_embedding.reset_parameters()
        self.reset_logit_scale()
        self.reset_channel_diagnostics()


class SenderTransformerLM(GumbelChannel, nn.Module):
    def __init__(
        self,
        referent_embedding_size,
        **kwargs
    ):
        """
        A Transformer speaker in two arms, selected by `bidirectional`: an
            autoregressive decoder (False) or Perceiver IO (True). See
            docs/architecture.md.

        https://arxiv.org/abs/2502.20604
        """
        super().__init__()
        self.referent_embedding_size = referent_embedding_size
        self.token_embedding_size = kwargs["token_embedding_size"]
        self.d_model = kwargs["d_model"]
        self.vocabulary = kwargs["vocabulary"]
        self.message_length = kwargs["message_length"]
        # The configured tau. The tau actually passed to `gumbel_softmax` is
        #     the `sampling_tau` property below, which tracks the learned scale.
        self.tau = kwargs["tau"]
        self.uniform_weight = kwargs["uniform_weight"]
        self.dropout = kwargs["dropout"]
        self.layers = kwargs["layers"]
        self.bidirectional = kwargs["bidirectional"]
        self.heads = kwargs["heads"]
        self.utility_tokens = kwargs["utility_tokens"]
        self.latent_message_multiplier = kwargs["latent_message_multiplier"]
        self.ff_inner_size = kwargs["ff_inner_size"]
        self.stochastic_depth = kwargs["stochastic_depth"]
        self.activation = model_util.get_activation(kwargs["activation"])
        self.relative_position_embedding = kwargs["relative_position_embedding"]
        self.pre_norm = kwargs["pre_norm"]
        self.post_norm = kwargs["post_norm"]
        self.return_bos_tokens = kwargs["return_bos_tokens"]
        self.knocking_heads = kwargs["knocking_heads"]
        self.depthwise_linear_stochastic_depth = kwargs[
            "depthwise_linear_stochastic_depth"
        ]
        self.ff_inner_dropout = kwargs["ff_inner_dropout"]
        self.ff_outer_dropout = kwargs["ff_outer_dropout"]
        self.self_attention_dropout = kwargs["self_attention_dropout"]
        self.cross_attention_dropout = kwargs["cross_attention_dropout"]
        # Resolved against this stack's own depth when the config asks for
        #     `"deepnorm"`. The two arms take different DeepNorm forms: the
        #     decoder arm cross-attends inside every block, the latent arm does
        #     not. See docs/broccoli.md.
        self.alpha, self.beta = model_util.resolve_residual_scaling(
            kwargs["alpha"],
            kwargs["beta"],
            kwargs["layers"],
            # The encoder form on both arms. Neither cross-attends from inside a
            #     block any more -- the referents arrive as the stack's *input*,
            #     not as a memory -- so both are two-branch stacks. See
            #     docs/broccoli.md.
            decoder=False,
        )

        self._init_channel(
            kwargs["init_energy"], self.vocabulary, self.uniform_weight
        )

        if self.referent_embedding_size != self.token_embedding_size:
            raise NotImplementedError(
                "`referent_embedding_size` and `token_embedding_size` must "
                "be the same for Transformer-based speaker models!"
            )

        if (
            self.utility_tokens
            and (int((self.d_model / self.heads) / self.utility_tokens) < 3)
        ):
            warnings.warn(
                "Fewer than 3 head dimensions per utility token may be suboptimal."
            )

        if self.message_length < 3:
            raise ValueError(
                "message_length must be at least 3 (due to SOS and EOS tokens)"
            )

        self.content_length = self.message_length - 2

        # The length of the latent array the stack runs over. The message is its
        #     *tail*, so this is the message's length plus however many free
        #     slots the multiplier buys above it. Those free slots are what the
        #     knob is for, and it is bandwidth they buy: the referents reach the
        #     language model through `heads * latent_length` softmax weights and
        #     nothing else. Rounded rather than floored so the knob is symmetric
        #     about the integers. See docs/architecture.md.
        self.latent_length = round(
            self.content_length * self.latent_message_multiplier
        )

        if self.latent_length < self.content_length:
            raise ValueError(
                "`latent_message_multiplier` "
                f"({self.latent_message_multiplier}) rounds the latent array to "
                f"{self.latent_length} positions at content length "
                f"{self.content_length}. The message is the tail of the latent "
                f"array, so the array cannot be shorter than the message: the "
                f"multiplier must be at least 1.0."
            )

        # Where the message begins. The slots before it are never overwritten by
        #     a sampled symbol, so every message slot reads them at every step --
        #     which is what a cross-attention memory used to do, folded into the
        #     self-attention branch. See docs/architecture.md.
        self.first_message_slot = self.latent_length - self.content_length

        # The encoder query: what to ask the prototypes, `latent_length` times.
        #     Built in both arms -- it is what turns two prototypes into an array
        #     wide enough to read from.
        self.query = nn.Parameter(
            torch.empty(self.latent_length, self.d_model)
        )

        self.query_layer_norm = nn.LayerNorm(self.d_model)
        self.referent_layer_norm = nn.LayerNorm(self.d_model)
        self.latent_layer_norm = nn.LayerNorm(self.d_model)

        if not self.bidirectional:
            # Causal arm only: the parallel arm never reads a symbol back.
            #     Sized at `d_model`, which the check above requires to equal
            #     `token_embedding_size` anyway. Both speakers feed the *soft*
            #     one-hot through this so the straight-through gradient reaches
            #     the step that produced the symbol.
            self.token_embedding = nn.Embedding(
                self.vocabulary + 4, # +4 for PAD, SOS, EOS, UNK
                self.d_model
            )

        # A learned tag marking which row of the prototype sequence is the
        #     positive concept and which is the negative one. Row 0 is positive,
        #     row 1 negative. Without it this speaker cannot read that order at
        #     all: the cross-attention below is bit-identical under swapping the
        #     two keys.
        #
        # Added after the norm, initialised as an antipodal pair at the
        #     normed prototype's own scale, and not otherwise scale-pinned; the
        #     name must keep "embedding" in it or `gradboard` will start decaying
        #     it. See docs/architecture.md for all four.
        self.polarity_embedding = nn.Parameter(
            torch.zeros(2, self.d_model)
        )

        # Every broccoli argument is set explicitly, including the inert ones.
        #     See docs/broccoli.md.
        self.cross_attention = broccoli.transformer.MHAttention(
            self.d_model,
            self.heads,
            # Attention internals get their own setting, never the speaker's
            #     `dropout`, which regularises the prototypes going in.
            dropout=self.cross_attention_dropout,
            causal=False, # Whole image informs whole latent array
            seq_len=self.latent_length,
            linear_module=nn.Linear,
            bos_tokens=0,
            knocking_heads=False,
            # No positional information here: the query carries its own order
            #     and the prototypes are an unordered pair. `positional_heads` is
            #     inert and pinned at the repo-wide 1.0; see docs/broccoli.md.
            rotary_embedding=None,
            positional_heads=1.0,
            source_size=None,
            scaling="d",
        )

        # One stack, both arms. `bidirectional` selects a mask and nothing else:
        #     the two arms build the same latent array from the same query, run
        #     the same blocks over it, and take the message from the same tail
        #     slots. The parallel arm reads all of them at once; the causal arm
        #     reads them one at a time, overwriting each with its symbol as it
        #     commits. See docs/architecture.md.
        #
        # This used to be a genuine fork -- a Perceiver IO encoder plus an output
        #     query on one side, a cross-attending `TransformerDecoder` on the
        #     other -- and `bidirectional` selected an architecture rather than a
        #     mask. That cost the causal arm its first symbol: its sequence began
        #     at SOS, a constant, so the concept reached symbol 0 only through
        #     cross-attention branches scaled by DeepNorm's `beta / alpha`. At
        #     init one seed in five then emitted the *same* first symbol for
        #     every concept. See docs/anecdotes.md.
        self.transformer = broccoli.transformer.TransformerEncoder(
            self.latent_length,
            self.d_model,
            self.layers,
            self.heads,
            # Pinned False, and no longer a config option; both arms run
            #     rotary. See docs/broccoli.md.
            absolute_position_embedding=False,
            relative_position_embedding=self.relative_position_embedding,
            # Pinned at 1.0 and no longer configurable; see docs/broccoli.md.
            positional_heads=1.0,
            # Derived, not configured: this block runs over the *latent*
            #     array. See `latent_length`.
            source_size=(self.latent_length,),
            # `ff_ratio` None so that `ff_inner_size` is the live knob; note
            #     broccoli's `ViT` resolves the two the other way round. See
            #     docs/broccoli.md.
            ff_ratio=None,
            ff_inner_size=self.ff_inner_size,
            activation=self.activation,
            activation_kwargs=None,
            ff_linear_module_up=None,
            ff_linear_module_down=None,
            # Pinned rather than promoted: this argument can never take
            #     effect. Use the inner/outer knobs. See docs/broccoli.md.
            ff_dropout=0.0,
            ff_inner_dropout=self.ff_inner_dropout,
            ff_outer_dropout=self.ff_outer_dropout,
            msa_dropout=self.self_attention_dropout,
            stochastic_depth=self.stochastic_depth,
            depthwise_linear_stochastic_depth=self.depthwise_linear_stochastic_depth,
            # The whole of what `bidirectional` now selects. False here means
            #     the array is a set and every slot reads every other; True
            #     means the message slots in its tail are generated in order,
            #     each reading only what came before it. Either way the free
            #     slots ahead of the message are visible from all of them.
            causal=not self.bidirectional,
            linear_module=nn.Linear,
            bos_tokens=self.utility_tokens,
            knocking_heads=self.knocking_heads,
            return_bos_tokens=self.return_bos_tokens,
            pre_norm=self.pre_norm,
            post_norm=self.post_norm,
            msa_scaling="d",
            alpha=self.alpha,
            beta=self.beta,
        )

        # Pre-norm token prior, as in `SenderGRULM`; see docs/channel.md.
        self.outputs2vocab = nn.Linear(
            self.d_model,
            self.vocabulary + 4 # +4 for PAD, SOS, EOS, UNK
        )

        self.reset_parameters()

    def encode(
        self,
        prototypes
    ):
        """
        Read the two prototypes into a latent array of `latent_length`
            positions. Shared by both arms.

        Returns:
            latents: (batch, latent_length, d_model), *unnormalised* -- the two
                arms want it normalised at different points.
        """
        batch_size = prototypes[0].size(0)

        stack_prototypes = torch.stack(prototypes, 1) # To sequence

        # Tag after the norm, not before -- see `polarity_embedding`. Shape
        #     (2, d_model) broadcasts over the batch.
        normed_prototypes = (
            self.referent_layer_norm(stack_prototypes) + self.polarity_embedding
        )

        query = self.query.unsqueeze(0).expand(
            batch_size,
            self.latent_length,
            self.d_model
        )

        normed_query = self.query_layer_norm(query)

        return self.cross_attention(
            normed_query, normed_prototypes, normed_prototypes
        ) # (batch, self.latent_length, self.d_model)

    def embeddings(
        self,
        prototypes
    ):
        """
        The parallel arm's whole forward pass: one vector per symbol slot, a
            function of the prototypes alone.

        Parallel arm only, and deliberately not given a causal-arm branch --
            there the embeddings depend on the symbols sampled before them, so
            no honest signature exists and the loop lives in `decode`.

        The message is the latent array's tail rather than a separate readout.
            There used to be an `output_query` here, `content_length` learned
            rows cross-attending the processed latents; taking the tail says the
            same thing with one parameter fewer and, more to the point, says it
            identically on both arms. What `latent_message_multiplier` buys is
            now free slots *ahead* of the message rather than a separately-sized
            array behind it.
        """
        latents = self.transformer(
            self.latent_layer_norm(self.encode(prototypes))
        )

        return latents[
            :, self.first_message_slot :, :
        ] # (batch, self.content_length, self.d_model)

    def record_polarity_separation(self):
        """
        `norm(e_pos - e_neg)`, the only part of `polarity_embedding` the
            cross-attention can act on, opening at exactly zero. A parameter
            norm rather than a per-batch quantity. See docs/measurement.md.
        """
        with torch.no_grad():
            self.polarity_separation = (
                self.polarity_embedding[0] - self.polarity_embedding[1]
            ).norm().item()

    def decode_autoregressively(
        self,
        prototypes
    ):
        """
        The causal arm's sampling loop: the latent array decoded in place, one
            tail slot at a time, each conditioned on the free slots ahead of the
            message and on the symbols already committed behind it.

        A step-for-step mirror of `SenderGRULM.decode` from the sampling
            onwards, and deliberately does not restate the reasoning behind the
            ordering there. What differs is that this threads the symbols
            themselves rather than a hidden state, re-reading the whole array
            through the stack at every step -- see docs/architecture.md for why
            re-reading is exact.

        The point of the overwrite. Slot `first_message_slot + i` holds a
            *concept-derived* vector when it is read, because `encode` put one
            there, so the residual stream that produces symbol `i` starts as the
            referent and DeepNorm's `alpha` amplifies it. That is what
            `SenderGRULM.init_h` does for the GRU. The sequence used to open at
            SOS -- a constant -- and the concept had to arrive through a
            cross-attention branch scaled by `beta / alpha`; docs/anecdotes.md
            has what that cost the first symbol.

        Under the causal mask nothing after slot `i` is visible from it, so the
            slots that have not been sampled yet may keep their latent vectors.
            Nothing reads them until they are overwritten.

        Returns: as `decode`.
        """
        batch_size = prototypes[0].size(0)
        device = prototypes[0].device

        # Normalised once, here, rather than inside each block. The array is the
        #     stack's input on this arm, and it shares the sequence with token
        #     embeddings, which arrive at their own scale.
        rows = list(
            self.latent_layer_norm(self.encode(prototypes)).unbind(1)
        ) # `latent_length` tensors of (B, D)

        lang = []
        symbol_embeddings = []
        survival_logits = []
        spread_steps = []

        lang.append(
            self.reserved_onehot(data.language.SOS_IDX, batch_size, device)
        )

        if self.training:
            self.record_polarity_separation()

        for i in range(self.content_length):
            slot = self.first_message_slot + i

            step_embedding = self.transformer(
                torch.stack(rows, dim=1)
            )[:, slot, :] # (B, D)
            symbol_embeddings.append(step_embedding)

            logits = self.outputs2vocab(step_embedding) # (B, V + 4)

            if self.training:
                spread_steps.append(self.logit_spread_of(logits))

            predicted_onehot, pre_gain = self.sample_symbols(logits)

            if self.training:
                survival_logits.append(pre_gain)

            lang.append(predicted_onehot.unsqueeze(1))

            # Through the embedding matrix rather than an index lookup: the
            #     sampled one-hot is soft in the backward pass, and a matmul is
            #     what lets the straight-through gradient reach the step that
            #     produced it.
            rows[slot] = predicted_onehot @ self.token_embedding.weight

        if self.training:
            # Pooled over positions after the loop, as in `SenderGRULM.decode`.
            self.record_survival(torch.stack(survival_logits, 1))
            self.logit_spread = torch.stack(spread_steps).mean().item()

        lang.append(
            self.reserved_onehot(data.language.EOS_IDX, batch_size, device)
        )

        return torch.cat(lang, 1), torch.stack(symbol_embeddings, 1)

    def decode(
        self,
        prototypes
    ):
        """
        Produce a message and the symbol embeddings that produced it, and
            dispatch on the arm.

        The parallel arm's embeddings are a function of the prototypes alone;
            the causal arm's depend on the symbols sampled before them, so there
            they correspond only within one call. Either way both come back
            together, so callers can treat all three speakers identically.

        Returns:
            onehot: (batch, message_length, vocabulary + 4)
            symbol_embeddings: (batch, message_length - 2, d_model), one dense
                contextual vector per content symbol, taken before the
                vocabulary projection and sampling
        """
        if not self.bidirectional:
            return self.decode_autoregressively(prototypes)

        batch_size = prototypes[0].size(0)
        device = prototypes[0].device

        symbol_embeddings = self.embeddings(prototypes)

        logits = self.outputs2vocab(symbol_embeddings)

        if self.training:
            self.logit_spread = self.logit_spread_of(logits).item()
            self.record_polarity_separation()

        onehot_content, pre_gain = self.sample_symbols(logits)

        if self.training:
            # This arm emits every position in one shot, so its logits are
            #     already pooled over positions.
            self.record_survival(pre_gain)

        onehot = torch.cat(
            [
                self.reserved_onehot(data.language.SOS_IDX, batch_size, device),
                onehot_content,
                self.reserved_onehot(data.language.EOS_IDX, batch_size, device),
            ],
            dim=1,
        )

        return onehot, symbol_embeddings

    def forward(
        self,
        prototypes,
        **kwargs
    ):
        return self.decode(prototypes)[0] # (batch, message_length, vocabulary)

    def reset_parameters(self):
        nn.init.normal_(self.query, mean=0.0, std=1.0)
        self.query_layer_norm.reset_parameters()
        self.referent_layer_norm.reset_parameters()
        self.latent_layer_norm.reset_parameters()
        positive_tag = torch.randn_like(self.polarity_embedding[0])
        with torch.no_grad():
            self.polarity_embedding.copy_(
                torch.stack([positive_tag, -positive_tag])
            )
        self.cross_attention.reset_parameters()
        self.transformer.reset_parameters()

        # The only module either arm has that the other does not: the causal arm
        #     reads its own symbols back, and the parallel arm never does.
        if not self.bidirectional:
            self.token_embedding.reset_parameters()

        self.outputs2vocab.reset_parameters()
        self.reset_logit_scale()
        self.reset_channel_diagnostics()


class Sender(nn.Module):
    def __init__(
        self,
        feat_model: nn.Module,
        prototyper: nn.Module,
        language_model: nn.Module,
        contrast: Optional[nn.Module] = None,
        vision_dropout: float= 0.5,
        prototype_dropout: float= 0.5
    ):
        """
        An agent that receives positive and negative examples of a concept and
            produces an utterance intended to communicate it.

        Args:
            feat_model: produces embeddings from referents
            prototyper: builds prototypes from positive and negative examples
            language_model: builds utterances from prototypes
            contrast: optional `ExampleContrast`, run between the two so the
                referents inform each other before they are pooled. `None`
                leaves the speaker exactly as it was.
            vision_dropout: dropout on per-image embeddings, before pooling
            prototype_dropout: dropout on the pooled concept vectors. This is
                where jayelm's single `--dropout` sits; the pre-pool mask is the
                weaker of the two. See docs/architecture.md.
        """
        super().__init__()
        self.feat_model = feat_model
        self.feat_size = feat_model.final_feat_dim
        self.prototyper = prototyper
        self.contrast = contrast
        self.language_model = language_model
        self.vision_dropout = nn.Dropout(p=vision_dropout)
        self.prototype_dropout = nn.Dropout(p=prototype_dropout)

    def embed_images(self, samples):
        """
        Embed every referent image in a batch, reshaping (batch, referents, ...)
            to a flat batch of images and back.
        """
        batch_size = samples.shape[0]
        n_obj = samples.shape[1]
        rest = samples.shape[2:]
        flat_samples = samples.view(batch_size * n_obj, *rest)
        embedded_samples = self.vision_dropout(self.feat_model(flat_samples))
        return embedded_samples.view(batch_size, n_obj, -1)

    def get_prototypes(self, samples, targets):
        if samples.size(1) % 2 != 0:
            raise NotImplementedError(
                "The prototyper must be passed an even number of samples, "
                "the first n / 2 should be positive and the rest negative."
            )
            
        midp = targets.shape[1] // 2

        if not ((targets[:, :midp] == 1.0).all() and (targets[:, midp:] == 0.0).all()):
            raise NotImplementedError(
                "The prototyper must be passed an even number of samples, "
                "the first n / 2 should be positive and the rest negative."
            )

        embedded = self.embed_images(samples)

        # Between the vision model and the pooling, so the contrast sees the
        #     same `vision_dropout` mask the prototyper does and what it returns
        #     is pooled exactly as the backbone's own output would have been.
        #     Guarded on `None` rather than by `hasattr`, matching
        #     `reset_parameters` below.
        if self.contrast is not None:
            embedded = self.contrast(embedded, targets)

        prototypes = self.prototyper(embedded, targets)

        # Applied per prototype rather than to the concatenation, which is the
        #     same thing: dropout masks each element independently.
        return tuple(self.prototype_dropout(p) for p in prototypes)

    def speak(self, samples, targets):
        """
        Produce a message, the symbol embeddings behind it, and the concepts
            that prompted it, all from a single pass -- which is what
            compositionality analysis needs. See docs/architecture.md.

        Returns:
            messages: (batch, message_length, vocabulary + 4)
            symbol_embeddings: (batch, message_length - 2, embedding size), one
                dense contextual vector per content symbol, positionally aligned
                with the output of `trim_messages`
            concepts: (batch, 2 * referent embedding size)
        """
        prototypes = self.get_prototypes(samples, targets)
        messages, symbol_embeddings = self.language_model.decode(prototypes)
        return messages, symbol_embeddings, torch.cat(prototypes, 1)

    def forward(
        self,
        samples,
        targets,
        **kwargs
    ):
        # Prototype once and reuse: a second `get_prototypes` call would re-run
        #     the vision model under a fresh `vision_dropout` mask.
        prototypes = self.get_prototypes(samples, targets)

        messages = self.language_model(
            prototypes,
            **kwargs
        )

        return messages, torch.cat(prototypes, 1)

    def reset_parameters(self):
        # No `hasattr` guard, matching `Receiver.reset_parameters`: a guard turns
        #     a missing method into a silently skipped backbone. See
        #     docs/anecdotes.md.
        self.feat_model.reset_parameters()
        self.prototyper.reset_parameters()
        # The `None` guard is the architecture, not a `hasattr` fallback: a
        #     speaker built without the contrast stage has nothing to reset,
        #     where a speaker built with one that had been renamed must raise.
        if self.contrast is not None:
            self.contrast.reset_parameters()
        self.language_model.reset_parameters()