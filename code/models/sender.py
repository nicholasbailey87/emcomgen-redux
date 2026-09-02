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

The channel -- `layer_norm_logits`, `logit_scale`, `uniform_weight`, the two
    gradient estimators and their diagnostics -- is documented in
    docs/channel.md.
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


# The speaker's channel scale is bounded above by projection rather than by a
#     `clamp` in the forward pass. See `GumbelChannel.project_channel`.
MAX_LOGIT_SCALE = 2.0


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


def sharpest_logit_margin(vocabulary: int) -> float:
    """
    The largest top-two gap `layer_norm_logits` permits, in units of the logits'
        own standard deviation.

    The normaliser pins the emittable logits to zero mean and unit variance, and
        the most concentrated arrangement satisfying both is one token at
        `sqrt(V - 1)` with the remaining `V - 1` at `-1/sqrt(V - 1)`: the mean is
        `(sqrt(V-1) - (V-1)/sqrt(V-1)) / V = 0` and the variance is
        `((V-1) + (V-1)/(V-1)) / V = 1`. Its margin is the sum of those two, which
        simplifies to `V / sqrt(V - 1)` -- 3.883 at V = 14, 4.588 at V = 20.

    This is a *hard* bound rather than a typical value, and it is why fidelity
        never depended on `logit_scale`: at a scale of *one* a speaker that
        spends its whole shape budget on one token already reaches 0.789 unmixed
        survival at V = 14. The scale moves that ceiling; the shape budget is
        what sets it. `logit_margin` is read against this number -- see
        `mean_logit_margin` and docs/measurement.md.

    Args:
        vocabulary: number of emittable tokens

    Returns:
        The margin, in standard deviations
    """
    return vocabulary / math.sqrt(vocabulary - 1)


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
    scale: float,
    uniform_weight: float,
) -> torch.Tensor:
    """
    The fraction of symbols that survive the Gumbel noise, averaged over slots,
        i.e. the `realised_survival` column.

    By the Gumbel-max identity this is just the winning token's softmax
        probability, so no Monte Carlo is needed; `tests/test_exploration.py`
        pins the identity. Applies the real pipeline in the real order -- scale
        first, then the uniform mixture -- so the mixture's bounds hold.

    Purely a measurement.

    Args:
        logits: (..., vocabulary + 4), reserved tokens already masked to -inf
        scale: the multiplier applied before mixing, i.e. `logit_scale`
        uniform_weight: as in `flatten_logit_distribution`

    Returns:
        A scalar tensor, the mean over all slots of the winning token's
            post-mixing probability
    """
    scaled = logits * scale
    if uniform_weight > 0.0:
        scaled = flatten_logit_distribution(scaled, uniform_weight)
    return scaled.softmax(-1).max(-1).values.mean()


def mean_logit_margin(logits: torch.Tensor) -> torch.Tensor:
    """
    The winning token's lead over the runner-up, averaged over slots, i.e. the
        `logit_margin` column.

    **Why this and not `logit_spread`.** Saturation is set by the margin times
        `logit_scale`, not by the scale alone: `1 - p` is about
        `(V - 1) * exp(-scale * margin)` for the winner's probability `p`.
        `layer_norm_logits` pins the emittable logits' *second moment* and pins
        nothing about their shape, so a speaker can saturate its channel shut
        without moving the scale at all, by growing this instead. That is what
        the 2026-08-29 ShapeWorld run did: at `logit_scale` 3.046 the winner sat
        at 0.99951, which inverts to a margin of ~3.35, against the ~0.44 that
        i.i.d. standard normal logits over V = 14 would give. Nothing in
        metrics.csv could see it -- `logit_spread` is the std of the *raw*
        logits, taken before the normaliser divides exactly that back out, so it
        reads the head's output magnitude and would report the same for a head
        that grew uniformly as for one that grew a spike.

        `logit_scale` learns again as of 2026-08-31, so the two routes are both
        open and this column is what separates them: a speaker sharpens by
        growing its margin, by growing its scale, or by both, and only the
        product is visible in `unmixed_survival`. `sharpest_logit_margin` is
        where the first stops -- 3.883 at V = 14 -- and `MAX_LOGIT_SCALE` is
        where the second does. Neither bound is there to protect the backward
        pass: under `estimator = "identity"` the Jacobian is `I` however sharp
        the speaker gets, which is why the scale no longer needs solving against
        a saturation ceiling. See docs/channel.md.

    Taken on the emittable slice only, and *after* `layer_norm_logits`, so the
        result is already in units of the logits' own standard deviation and
        needs no further scaling to compare across runs or vocabularies. Pass
        the same tensor `record_survival` is given.

    Purely a measurement, like `mean_winning_probability` above.

    Args:
        logits: (..., vocabulary + 4), normalised, reserved tokens first

    Returns:
        A scalar tensor, the mean over all slots of the top-two gap
    """
    top_two = logits[..., 4:].topk(2, dim=-1).values
    return (top_two[..., 0] - top_two[..., 1]).mean()


def logit_prior_share(logits: torch.Tensor) -> torch.Tensor:
    """
    The fraction of the normalised logits' variance that is the *same for every
        input*, i.e. the `logit_prior_share` column.

    **The question no other channel column answers.** `layer_norm_logits` pins
        each position's emittable logits to unit variance, so the speaker's
        whole shape budget is fixed and the only question is what it spends it
        on. Concentrating that budget on one token is not a failure -- a sharply
        peaked distribution whose peak *moves with the input* is a perfect
        channel, confident and informative. The failure is peaking on the same
        token whatever it saw, which is confidence with zero information, and
        `realised_survival`, `logit_scale` and `logit_margin` all read
        identically in the two cases.

        This separates them. Split the logits over the batch into the component
        common to every input and the residual that varies: the two are
        orthogonal, so the shares sum to 1. Near 0 the shape is entirely
        input-driven. At 1 the speaker emits one message for every game.

    **Why the shape budget is worth watching at all.** The most concentrated
        distribution the normaliser permits is one token at `sqrt(V-1)` and the
        rest at `-1/sqrt(V-1)` -- a margin of 3.883 sd at V = 14, which reaches
        unmixed survival 0.789 at a `logit_scale` of *one*. The 2026-08-29
        ShapeWorld run had used 86% of that budget. So fidelity does not have to
        come from the scale and never did -- which is why the scale is now free
        to learn rather than solved against a saturation ceiling it was the only
        assumed route to. Shape spent on confidence is still shape not spent on
        meaning, and that this column is what says so is unchanged.

    The cheapest way to buy input-independent shape is `outputs2vocab.bias`,
        which sits before the normaliser and so survives it as a fixed pattern,
        and which is on the weight-decay exclusion list. Read this column
        against that parameter.

    Purely a measurement, like the two above.

    Args:
        logits: (batch, ..., vocabulary + 4), normalised, reserved tokens first

    Returns:
        A scalar tensor in [0, 1], or NaN for a batch of one, where every input
            is trivially its own mean and the share is 1 by construction
    """
    emittable = logits[..., 4:]

    if emittable.size(0) < 2:
        return torch.full((), float("nan"), device=logits.device)

    common = emittable.mean(0, keepdim=True).expand_as(emittable)
    total = emittable.pow(2).sum()

    return common.pow(2).sum() / total if total > 0.0 else torch.zeros(())


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

        The tag rides the queries and keys only; the values are the untagged
        referents. Polarity chooses what a query reads and is absent from what
        comes back, which is what stops the stage collapsing into a learned
        per-polarity offset. See `forward`.
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
        adapted = self.layer_norm(self.adapter(samples))

        # Tagged as queries and keys, untagged as values. Polarity decides which
        #     referents a query reads and is deliberately absent from what comes
        #     back. With the tag in the values too, every output carries
        #     `(sum_pos a - sum_neg a) * V(tag)` -- a free learned per-polarity
        #     vector, which is the cheapest thing this stage can produce and is
        #     not contrast between examples at all. `contrast_within_share` is
        #     the column that reads it: 0.50-0.67 on the rung 10 run that
        #     learned, against 0.0034 on the 2026-08-26 one that did not, where
        #     99.7% of the branch was a common vector plus that offset.
        #
        # The addressing survives, which is the case the stage exists for: a
        #     positive query finds the negatives through their keys and gets
        #     their *content* back to compare itself against, rather than a flag
        #     saying "negative". A polarity-shaped output can still emerge --
        #     a query attending only to positives returns the positive mean --
        #     but it is then built from content rather than from a parameter.
        tagged = adapted + tag
        branch = self.out_projection(self.attention(tagged, tagged, adapted))
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
        #     is the example count and there is no scoring vector to report --
        #     neither its norm nor the spread of the scores it would produce.
        self.pool_effective_examples = float("nan")
        self.pool_score_norm = float("nan")
        self.pool_score_sd = float("nan")

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
        self.pool_score_sd = float("nan")

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
        """
        `pool_effective_examples` compresses; `pool_score_sd` does not.

        The effective count is `1 / sum(w^2)`, which for a score spread `sigma`
            is about `n / (1 + sigma^2)` -- so the whole interval from "barely
            structured" to "perfectly uniform" is squeezed into the last
            fraction of a percent below `n`. On the 2026-08-29 ShapeWorld run
            9.86 and 9.996 differ by 1.3% in this column and by *sixfold* in the
            underlying spread, and that was the difference between a collapse
            the speaker recovered from at epoch 17 and the one it did not at
            epoch 21. Both columns are kept: the count is the interpretable one
            and this is the one with resolution where it matters.
        """
        weights = torch.cat([positive_weights, negative_weights])

        self.pool_effective_examples = (1.0 / weights.pow(2).sum(-1)).mean().item()

        # The scores themselves, recovered rather than recomputed: softmax makes
        #     `log w = s - logsumexp(s)`, an additive constant per game, so the
        #     standard deviation over examples is the scores' own exactly. That
        #     avoids reaching past `SequencePool.attention_scores` into the
        #     `Sequential` it is built from, whose layout is broccoli's to
        #     change. `float` first: under autocast the weights arrive in fp16
        #     and the log of a small one loses most of its digits there.
        #
        # Clamped because the recovery is only exact while the softmax has not
        #     underflowed. Past a score gap of about 87 nats a loser's weight is
        #     0 in fp32, `log` gives -inf and the standard deviation is NaN --
        #     which would read as "no pooler" rather than as the total
        #     commitment it is. The floor turns that into a large finite number
        #     instead, and it is unreachable in any regime the column is
        #     informative in: the largest spread observed on a live run is 0.93.
        floor = torch.finfo(torch.float32).tiny
        self.pool_score_sd = (
            weights.float().clamp_min(floor).log().std(-1).mean().item()
        )

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

    A mixin rather than a submodule, so `log_logit_scale` stays registered on
        the speaker's language model itself: the `state_dict` key and the
        `named_parameters` suffix are the ones `split_out_parameter` and
        `SCALAR_GROUPS` match on. Exactly the arrangement `receiver.ScoreVolume`
        uses for the listener's volume, which is the scalar this one is the
        counterpart of.

    The channel is that one parameter, an estimator name settled at
        construction, and the per-batch diagnostics.
    """

    def _init_channel(self, estimator):
        """
        Call from `__init__` where the parameter should be created: creation
            order fixes which RNG draw every later parameter gets. `torch.zeros`
            does not consume the generator, so adding this one back in 2026-08-31
            left every other parameter's draw at a given seed exactly where
            `44767b2` had put it. What it does change is `state_dict`:
            checkpoints written between those two commits have no
            `log_logit_scale` key and will not load.

        Stored as its log so `exp` keeps it strictly positive, as
            `ScoreVolume.log_score_scale` is and for the same reason: gradient
            descent cannot walk a scale through zero and out the far side.
            Opens at 1.0, where `layer_norm_logits`' shape budget alone already
            reaches 0.789 unmixed survival at V = 14.

        There is no floor. A speaker with nothing to say is pushed flatter --
            docs/channel.md records the old parameter sliding 0.9094 -> 0.6547 on
            rung 10 -- and that is self-regulation rather than a failure mode,
            because `scale_without_attenuating` means a small scale is a noisy
            channel and not a starved one. The ceiling is `MAX_LOGIT_SCALE`, and
            it is applied by `project_channel` after the optimiser step rather
            than by a `clamp` in `forward`. See that method.
        """
        self.log_logit_scale = nn.Parameter(torch.zeros(()))
        self.estimator = estimator

        self.reset_channel_diagnostics()

    @property
    def logit_scale(self):
        """
        The multiplier applied to the normalised logits, always positive. Read
            here rather than exponentiating at the use site so the sampler and
            the metrics column cannot drift apart -- `ScoreVolume.score_scale`
            is the same property for the same reason.
        """
        return self.log_logit_scale.exp()

    def project_channel(self):
        """
        Bound the scale above at `MAX_LOGIT_SCALE`. Called from `train.py`'s
            `optimiser_step`, after the step.

        **Why projection and not a `clamp` in the forward pass.** The rule
            `receiver.py` states about the mix floor holds here: `clamp`'s
            gradient is zero *past* the bound and is not directional, so a
            parameter that overshoots gets no gradient in either direction and
            welds there permanently. `weight_decay` is 0.0, so nothing else
            would pull it back. `2 * sigmoid(x)` avoids the weld but its
            derivative `scale(1 - scale/2)` saturates at *both* ends, making
            recovery from a low scale about 5x slower than the descent that got
            there; under `exp` the rate is proportional, which is the right
            behaviour for a scale. Projecting after the step keeps `exp`'s
            proportional traverse and an exact 2.0, and the gradient stays live
            right up to the bound -- sitting at the ceiling costs nothing and
            leaving it is free.

        The cost is that the constraint is applied outside `forward`. That cost
            is paid here, as a method, so the module still owns the rule.

        `scaler.step` may skip the step entirely on inf/nan; this is idempotent,
            so that is harmless.
        """
        with torch.no_grad():
            self.log_logit_scale.clamp_(max=math.log(MAX_LOGIT_SCALE))

    def reset_channel_scale(self):
        """
        Back to the opening 1.0. Separate from `reset_channel_diagnostics` so a
            `reset_parameters` that wants one is not forced to take the other,
            exactly as `ScoreVolume.reset_score_volume` is separate.
        """
        with torch.no_grad():
            self.log_logit_scale.zero_()

    def reset_channel_diagnostics(self):
        """
        Per-batch diagnostics for metrics.csv. On both speakers so the two write
            the same columns; `polarity_separation` is not-applicable rather
            than unused on the GRU. See docs/measurement.md.
        """
        self.realised_survival = float("nan")
        self.unmixed_survival = float("nan")
        self.logit_margin = float("nan")
        self.logit_prior_share = float("nan")
        self.logit_spread = float("nan")
        self.polarity_separation = float("nan")

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

    def _gumbel_sample(self, normalised):
        """
        The forward sampler, shared by both estimators so that neither can drift
            from the other: at the same seed a gumbel run and an identity run
            emit *identical* messages, which is what makes an A/B between them a
            control rather than two different runs.

        The order is not interchangeable. Scaling the *masked* logits rather
            than re-masking after the scale sends `-inf` into the arithmetic and
            NaN out of it. See docs/channel.md.

        `hard=True` emits `argmax(logits + g)` with `g ~ Gumbel(0, 1)`, so the
            symbol is invariant to `tau`: `tau` shapes the soft surrogate the
            gumbel estimator differentiates and nothing else, and on the identity
            branch -- which discards that surrogate -- it does nothing at all.

        Args:
            normalised: (..., vocabulary + 4) from `layer_norm_logits`, reserved
                tokens not yet masked

        Returns:
            A hard one-hot of the same shape
        """
        # Through `scale_without_attenuating`, restoring what `7b10d47` did
        #     before the parameter was deleted: the forward value is
        #     `logit_scale * normalised` exactly as it reads, but
        #     `d/dnormalised` is 1 rather than `logit_scale`, so the scale's
        #     value never multiplies the speaker's whole stack. The scale keeps
        #     its own true partial and so is as free to slide as it ever was.
        scaled = mask_reserved_tokens(
            model_util.scale_without_attenuating(normalised, self.logit_scale)
        )

        if self.uniform_weight > 0.0:
            scaled = flatten_logit_distribution(scaled, self.uniform_weight)

        return F.gumbel_softmax(scaled, tau=self.tau, hard=True, dim=-1)

    def sample_symbols(self, logits):
        """
        Turn one step's (or one message's) logits into symbols, under whichever
            gradient estimator `sender_language_model.estimator` names.

        Returns `(onehot, pre_gain_logits)` -- the masked, normalised, *un*scaled
            logits the survival diagnostic is measured from, or None outside
            training. Callers pool it over positions themselves.

        **`"gumbel"`** is the estimator this model has always used: the hard
            one-hot forward, and backward through the soft sample
            `gumbel_softmax` builds on the way, whose Jacobian is
            `diag(p) - p pT`.

        **`"identity"`** keeps that forward exactly and replaces the backward
            with `I`:

                y = onehot.detach() + (z - z.detach())

            The sample is unchanged and still faithful -- `argmax(z + g)` *is* a
            categorical draw from `softmax(z)` -- so this changes what the
            speaker learns from, not what it says.

        **Why.** The soft Jacobian's cost is its *rank*, not its size. The
            per-token gradients are summed into one vector before they reach the
            language model and the vision trunk, and `diag(p) - p pT` at
            `p ~ onehot` has rank ~1, so all but one direction is destroyed
            before any optimiser or clipper sees it and the trunk hears a single
            token's opinion. Magnitude, by contrast, largely cancels: AdamW
            updates by `m / sqrt(v)`, and `clip_gradients` renormalises what
            survives. Only the identity estimator removes the rank collapse.
            Under it the speaker's gradient is `dL/dy` -- the receiver's
            per-token embedding sensitivity -- which is full rank and the same
            size however sharp the speaker has become. That is also why
            `logit_scale` is free to learn again: with `I` in the backward pass
            there is no saturation for a climbing scale to shut, so bounding `p`
            stopped being a gradient safeguard.

        **The surrogate is built on the emittable slice.** `masked` holds `-inf`
            in the four reserved columns and `-inf - (-inf)` is NaN. Slicing also
            stops those columns receiving gradient at all: they are constants
            from the sampler's point of view, so `outputs2vocab` rows 0-3 and the
            stack behind them are never trained toward tokens that cannot be
            emitted.

        **It taps the *scaled* logits**, and has to: `_gumbel_sample` runs
            under `no_grad` on this branch, so the surrogate is the only path
            back and an unscaled tap would leave `log_logit_scale` with no
            gradient at all. The composition is what makes that free. With
            `d(scaled)/d(normalised) = 1` from `scale_without_attenuating` and
            `dy/d(scaled) = I` from the estimator:

                dL/dnormalised   = dL/dy
                dL/dlog_logit_scale = <dL/dy, normalised> * logit_scale

            The first is bit-identical to what the unscaled tap gave, so the
            speaker's stack sees exactly the gradient it saw before the scale
            came back; the second is real and nonzero. See docs/channel.md.
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

        if self.estimator == "identity":
            # No graph through the sampler at all: the surrogate below is the
            #     only path back to the speaker.
            with torch.no_grad():
                onehot = self._gumbel_sample(normalised)

            # The bracketing is load-bearing. `onehot + z - z.detach()`
            #     associates left, so it computes `(1 + z) - z`, which in
            #     float32 is 1.0000001 rather than 1 -- a real perturbation of
            #     the message, on the winning token, every step. Forming the
            #     zero first makes the addition exact.
            emittable = model_util.scale_without_attenuating(
                normalised[..., 4:], self.logit_scale
            )
            onehot = torch.cat(
                [
                    onehot[..., :4],
                    onehot[..., 4:] + (emittable - emittable.detach()),
                ],
                dim=-1,
            )
        else:
            onehot = self._gumbel_sample(normalised)

        return onehot, masked.detach()

    def record_survival(self, pre_gain_logits):
        """
        Four columns off one tensor: the channel's fidelity with the uniform
            mixture applied, the same thing without it, and the two shape
            readings that say what the fidelity was bought with.

        `unmixed_survival` is the quantity `estimator = "gumbel"` differentiates
            through. That branch takes the soft sample's Jacobian
            `diag(p) - p pT`, and `flatten_logit_distribution` is a convex
            mixture in probability space, so the estimator's Jacobian is
            `(1 - w)(diag(p) - p pT)` in the winner's probability *before* the
            mixture. The mixture hides it: `realised_survival` is capped at
            `(1 - w) + w / V`, which is 0.90714 at the ShapeWorld default, and
            a run pinned against that cap reads as 0.9067 while the probability
            that actually shapes the gradient is 0.99951. Two orders of
            magnitude, invisible in the mixed column. Same function, same order
            of operations, mixture off.

            Under `estimator = "identity"` it leaves the backward pass
            altogether: that branch's Jacobian is `I` whatever `p` reads. It is
            still worth watching there, as the channel's fidelity, but it is no
            longer a gradient diagnostic -- which is why nothing bounds it any
            more. `MAX_LOGIT_SCALE` and `sharpest_logit_margin` bound the two
            things it is bought with, and their product is what this column
            reads.

        `logit_margin` and `logit_prior_share` are the shape pair. The first is
            how concentrated the distribution is, the second how much of that
            concentration is the same whatever the speaker saw. A peaked
            distribution whose peak *moves* is a perfect channel; one that
            peaks on the same token every time is confidence with no
            information, and the survival columns read identically in the two
            cases.
        """
        detached = pre_gain_logits.float()

        # `float()` rather than the tensor: `mean_winning_probability` is typed
        #     for a plain scale and is pure measurement, and a live parameter
        #     here would build a graph off the detached diagnostic path.
        scale = float(self.logit_scale)

        self.realised_survival = mean_winning_probability(
            detached, scale, self.uniform_weight
        ).item()
        self.unmixed_survival = mean_winning_probability(
            detached, scale, 0.0
        ).item()
        self.logit_margin = mean_logit_margin(detached).item()
        self.logit_prior_share = logit_prior_share(detached).item()


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
        # Gradient shaping only: the emitted symbol is an argmax and so is
        #     invariant to it, and `estimator = "identity"` discards the soft
        #     sample it shapes. See `sample_symbols`.
        self.tau = kwargs["tau"]
        self.uniform_weight = kwargs["uniform_weight"]
        self.dropout = kwargs["dropout"]
        self.layers = kwargs["layers"]
        self.bidirectional = kwargs["bidirectional"]
        self.directions = 2 if self.bidirectional else 1

        self._init_channel(kwargs["estimator"])

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
        self.reset_channel_scale()
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
        # Gradient shaping only: the emitted symbol is an argmax and so is
        #     invariant to it, and `estimator = "identity"` discards the soft
        #     sample it shapes. See `sample_symbols`.
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

        self._init_channel(kwargs["estimator"])

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
        # Two independent draws rather than an antipodal pair. Only
        #     `e_pos - e_neg` reaches the cross-attention -- the two prototypes
        #     are its whole key/value sequence, so softmax annihilates whatever
        #     the rows share and the value path returns it as a constant -- and
        #     an independent pair is exactly an antipodal one along
        #     `(t_pos - t_neg) / 2` plus that constant. So this is a magnitude
        #     change and not a geometry one: the opening separation goes from
        #     `2 * sqrt(d_model)` = 35.8 to `sqrt(2 * d_model)` = 25.3, against
        #     the 13.19 the one learning rung 10 settled at from a zero init.
        #     docs/architecture.md called the antipodal opening a 2.7x
        #     overshoot; this halves the overshoot. `polarity_separation` still
        #     reads the difference correctly, but the common mode it now has is
        #     not logged anywhere.
        with torch.no_grad():
            self.polarity_embedding.normal_(mean=0.0, std=1.0)
        self.cross_attention.reset_parameters()
        self.transformer.reset_parameters()

        # The only module either arm has that the other does not: the causal arm
        #     reads its own symbols back, and the parallel arm never does.
        if not self.bidirectional:
            self.token_embedding.reset_parameters()

        self.outputs2vocab.reset_parameters()
        self.reset_channel_scale()
        self.reset_channel_diagnostics()


class Sender(nn.Module):
    def __init__(
        self,
        feat_model: nn.Module,
        adapter: nn.Module,
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
            adapter: `ReferentAdapter`, the constant stage that brings the
                backbone's output to the language model's `d_model`. Everything
                downstream of it -- the contrast stage, the prototyper, the
                language model -- is sized from its output rather than from the
                backbone's.
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
        # The width every stage after the backbone runs at. This is the
        #     adapter's output, not `feat_model.final_feat_dim`: the backbone
        #     no longer sets the speaker's width. See
        #     `model_util.ReferentAdapter`.
        self.adapter = adapter
        self.feat_size = adapter.output_features
        self.prototyper = prototyper
        self.contrast = contrast
        self.language_model = language_model
        self.vision_dropout = nn.Dropout(p=vision_dropout)
        self.prototype_dropout = nn.Dropout(p=prototype_dropout)

        # Per-batch diagnostics, read by `train.py` for metrics.csv. On the
        #     agent rather than on a submodule because the pair brackets the
        #     contrast stage and so belongs to neither side of it. See
        #     `_record_referent_spread` and docs/measurement.md.
        self.referent_spread = float("nan")
        self.referent_spread_backbone = float("nan")

    @torch.no_grad()
    def _record_referent_spread(self, embedded, attribute):
        """
        How much the referents within one polarity still differ from each other,
            relative to what they share.

        **What it is for.** `AttentionPrototyper` scores each example and pools
            by softmax, and `pool_score_sd` reads the spread of those scores --
            but a flat score can mean two different things, and they call for
            opposite fixes: the examples genuinely collapsed onto one point, or
            the single scoring direction rotated somewhere they do not vary.
            This column separates them. It measures the referents themselves and
            never touches the scoring vector, so a collapse here is the backbone
            and a flat `pool_score_sd` with this holding up is the pool.

        **The decomposition** is `ExampleContrast._record_diagnostics`', so the
            two are read on the same basis: subtract each polarity's own mean,
            then take the RMS of what is left over the RMS of the means
            themselves. Within a polarity because that is the unit the
            prototyper pools over -- a positive/negative difference is signal,
            not spread -- and as a ratio so it is dimensionless and unmoved by a
            global rescale of the embeddings.

        It falls when a vector *common* to the examples grows, which is the
            second thing worth catching: `contrast_share` reached 0.32 on the
            2026-08-29 run while `contrast_within_share` was 1.6e-4, a branch
            that was 99.98% a shared vector plus a per-polarity offset. Taken
            either side of that stage, the pair says whether the contrast branch
            is the thing doing the homogenising.

        Args:
            embedded: (batch, 2n, embedding size), positives first
            attribute: the name to write the result to
        """
        n_positive = embedded.size(1) // 2
        positive, negative = embedded[:, :n_positive], embedded[:, n_positive:]

        polarity_means = torch.cat(
            (
                positive.mean(1, keepdim=True).expand_as(positive),
                negative.mean(1, keepdim=True).expand_as(negative),
            ),
            dim=1,
        )

        common = polarity_means.float().pow(2).mean().sqrt()
        within = (embedded - polarity_means).float().pow(2).mean().sqrt()

        setattr(
            self,
            attribute,
            (within / common).item() if common > 0.0 else 0.0,
        )

    def embed_images(self, samples):
        """
        Embed every referent image in a batch, reshaping (batch, referents, ...)
            to a flat batch of images and back.
        """
        batch_size = samples.shape[0]
        n_obj = samples.shape[1]
        rest = samples.shape[2:]
        flat_samples = samples.view(batch_size * n_obj, *rest)
        # Adapter before the dropout, so the mask lands on the referent
        #     embedding the rest of the speaker actually reads -- the contrast
        #     stage relies on seeing the same mask the prototyper does, and
        #     `vision_dropout` is documented as being on per-image embeddings.
        adapted = self.adapter(self.feat_model(flat_samples))
        embedded_samples = self.vision_dropout(adapted)
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

        # Train pass only, matching every other diagnostic on the speaker.
        if self.training:
            self._record_referent_spread(embedded, "referent_spread_backbone")

        # Between the vision model and the pooling, so the contrast sees the
        #     same `vision_dropout` mask the prototyper does and what it returns
        #     is pooled exactly as the backbone's own output would have been.
        #     Guarded on `None` rather than by `hasattr`, matching
        #     `reset_parameters` below.
        if self.contrast is not None:
            embedded = self.contrast(embedded, targets)

        # After the contrast stage, so this is what the prototyper actually
        #     pools. Recorded unconditionally rather than only when the stage is
        #     on, so the two columns are equal on a rung without it rather than
        #     one of them being NaN -- equality is the informative reading.
        if self.training:
            self._record_referent_spread(embedded, "referent_spread")

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
        self.adapter.reset_parameters()
        self.prototyper.reset_parameters()
        # The `None` guard is the architecture, not a `hasattr` fallback: a
        #     speaker built without the contrast stage has nothing to reset,
        #     where a speaker built with one that had been renamed must raise.
        if self.contrast is not None:
            self.contrast.reset_parameters()
        self.language_model.reset_parameters()