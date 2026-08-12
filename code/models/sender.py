"""
Speaker models. This includes speakers with a GRU-based language model as
    originally presented in "Emergent Communication of Generalizations"
    (https://arxiv.org/abs/2106.02668) and speakers with causal or non-causal
    Transformer language models. The intention is to show that
    Transformer-based speakers can be just as successful in tasks and show
    equal or greater compositionality.
"""

import math
import warnings

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
    leading SOS and truncate at the first EOS. The language model masks the
    reserved tokens (PAD/SOS/EOS/UNK) mid-sequence, so the surviving tokens are
    all real symbols. Accepts anything iterable-of-iterables (e.g. a CPU tensor
    via ``.tolist()``); returns a list of python int lists.
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


# Well below `F.layer_norm`'s 1e-5 default, so that the normaliser keeps
#     normalising as a speaker's logit scale collapses rather than handing the
#     channel a quietly weaker scale. Safe in fp32, which is where autocast runs
#     LayerNorm anyway, and a genuinely constant logit vector still yields zeros
#     rather than NaN. See the note in `layer_norm_logits`.
LAYER_NORM_EPS = 1e-12


def layer_norm_logits(logits: torch.Tensor, vocabulary: int) -> torch.Tensor:
    """
    Normalise the *emittable* vocabulary logits to zero mean and unit variance,
        per example and per position.

    Only the last `vocabulary` columns are normalised. The leading four reserved
        slots (PAD/SOS/EOS/UNK) are concatenated back untouched: they are masked
        to -inf immediately afterwards so their values are irrelevant, but they
        must not be allowed to pollute the mean and standard deviation of the
        tokens that can actually be emitted.

    This replaces an `nn.BatchNorm1d` over the same columns. LayerNorm is the
        right normaliser here because the property wanted is that every speaker
        arrives at the exploration gain with logits of comparable *magnitude*,
        and LayerNorm delivers that per example rather than on average over a
        batch. It is also position-invariant for both speakers by construction
        (BatchNorm annihilated per-position offsets in the GRU, which sees one
        position per call, but preserved them in the Transformer, which sees all
        of them at once), has no running statistics so train and eval agree, and
        does not couple to `accumulator_steps`.

    Functional rather than a module, so that the affine parameters are
        structurally absent rather than merely disabled, and nothing is added to
        the `state_dict`. Without affine the transform is argmax-preserving, so
        it changes no eval-time message.

    `eps` is 1e-12 rather than the 1e-5 default, and that is load-bearing.
        `F.layer_norm` divides by `sqrt(var + eps)`, so scale invariance holds
        only while the incoming variance is large against `eps`; below that the
        normaliser quietly stops normalising and the emittable logits come out
        *smaller* than unit variance. Since `logit_scale` is a constant it
        cannot absorb that, where the per-batch solve it replaced could and
        silently did.

    This is not hypothetical, and the headroom is much smaller than the raw
        logit scales quoted in 1510a55 suggest. A freshly built GRU speaker
        emits pre-norm logits with a standard deviation of ~0.24, not the 1 to
        159 measured on unnormalised ladder arms, and at the 1e-5 default the
        normaliser starts giving out below ~0.01 — a margin of roughly 24x, not
        the three orders of magnitude previously claimed here. Shrinking that
        speaker's output layer 1000x drops realised survival from 0.43 to 0.09;
        a channel that noisy then starves the gradient that would restore the
        logits, so it runs away. Observed on a birds run whose
        `realised_survival` fell 0.47 -> 0.17 over 22 epochs.

    At 1e-12 the same 1000x collapse leaves survival at 0.43, unchanged to four
        decimal places, and the normaliser holds down to a standard deviation of
        ~1e-6. `tests/test_exploration.py` pins both the invariance and where it
        finally stops. `logit_spread` in metrics.csv is the column that makes a
        collapse visible rather than something inferred after the fact.

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


def logit_scale(coefficient: float, vocabulary: int) -> float:
    """
    The constant the normalised logits are multiplied by before sampling.

    This is the exploration control. `F.gumbel_softmax(..., hard=True)` emits
        `argmax(logits + g)` with `g ~ Gumbel(0, 1)`, whose standard deviation is
        a fixed 1.283, so what fraction of symbols survive the noise is set by
        the size of the logits relative to that. `layer_norm_logits` pins them to
        unit variance for every speaker, and this then says what that unit is
        worth.

    Why `ln(vocabulary)` and not a bare constant: a symbol survives only if its
        logit margin beats the largest of `V` independent Gumbel draws, and
        `E[max_i g_i] = ln V + gamma`. So the noise a winner must clear grows
        logarithmically in the vocabulary while LayerNorm holds the logits at
        O(1) regardless of it. Without the `ln V` term the same coefficient would
        mean materially different channels for ShapeWorld and birds: at a flat
        scale of 2.0 a freshly initialised speaker flips 50% of its symbols at
        V=14 and 56% at V=20, and 76% by V=128.

    With the term, the coefficient that fixes a given initial operating point is
        near enough constant -- within ~3% over V = 14..128, though V=8 wants
        ~8% less -- so `coefficient` is the only free parameter. At the default
        0.66 the scale is 1.74 for ShapeWorld's V=14 and 1.98 for birds' V=20,
        and both start a fresh speaker at ~54% of symbols flipped.

    Why a constant at all, rather than a per-batch solve against a requested
        rate: LayerNorm already equalises logit *scale* across architectures,
        which was the problem the solve existed for. Solving on top of it also
        pins the *shape* — it overwrites the speaker's own confidence in both
        directions to hold a target — and it does so hardest at initialisation,
        where a speaker's argmax is nearly input-independent and forcing 90%
        fidelity onto it means emitting one message for every input, confidently,
        from the first batch. A constant leaves the speaker free to start
        uncertain and earn confidence by learning a peaked distribution.

    Args:
        coefficient: `logit_scale_coefficient` from the config
        vocabulary: number of emittable tokens

    Returns:
        The multiplier, a plain float — it is a constant, so unlike the gain it
            replaced there is no buffer and nothing to checkpoint.
    """
    return coefficient * math.log(vocabulary)


def mask_reserved_tokens(logits: torch.Tensor) -> torch.Tensor:
    """
    Set the four reserved tokens (PAD/SOS/EOS/UNK) to -inf so they can never be
        emitted mid-message. SOS and EOS are attached by the caller instead, so
        messages are fixed-length.

    Out of place, because this now runs directly on the output of the vocabulary
        projection (or of batch norm), and writing -inf into that in place would
        be modifying a tensor autograd still needs.

    Runs before the exploration noise so that the uniform mixture is spread over
        the emittable tokens only — see `flatten_logit_distribution`.

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
    Returns a weighted average of

    The uniform component is spread over the emittable tokens only, i.e. those
        not already masked to -inf by `mask_reserved_tokens`. Spreading it over
        all `vocabulary + 4` slots and masking afterwards would throw away the
        4/(V+4) of it that landed on reserved tokens, so a nominal weight of
        0.1 would deliver 0.078. Masked positions stay masked.

    Args:
        logits: some provided unnormalised log probabilities
        uniform_weight: the relative weight to give the uniform distribution
            when mixing it in to the provided logits

    Returns:
        A torch.Tensor of logits where the absolute differences between
            logits is reduced - i..e. a less "certain" distribution
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
    #     mask is restored afterwards. `torch.where` routes the gradient to the
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
    The fraction of symbols that survive the Gumbel noise, averaged over slots.

    `F.gumbel_softmax(logits, hard=True)` emits `argmax(logits + g)` with
        `g ~ Gumbel(0, 1)`, and by the Gumbel-max identity the probability that
        slot's arg-max is unchanged by the noise is exactly the winning token's
        softmax probability. So survival can be read straight off a softmax: no
        Monte Carlo over noise draws, no assumed logit distribution, and no
        seed. `tests/test_exploration.py` pins the identity.

    Applies the real sampling pipeline in the real order — scale first, then the
        uniform mixture — so that the mixture's bounds hold. Mixing first and
        scaling afterwards destroys them.

    This is now purely a measurement. It used to be the inner loop of a solve
        that chose the scale to hit a requested rate; the scale is a constant,
        so what this reports is what the speaker's own logit *shape* buys it at
        that constant. Logged per epoch as `realised_survival`, and it is
        expected to move over a run rather than sit on a target.

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


class AveragePrototyper(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, samples, labels=None):
        """
        Args:
            samples: a tensor of shape (batch, n_examples, embedding size)
                where each example is an embedded image. n_examples must
                be even. The first n_examples / 2 examples are positive
                examples and the remainder are negative examples.
            labels: the labels for the examples provided. This argument
                exists for backwards compatibility, but is not used for
                anything as the first half of provided examples is always
                positive and the second half negative. See `samples` definition.
        """
        n_pos_ex = samples.size(1) // 2

        positive_examples = samples[:, :n_pos_ex, :]
        negative_examples = samples[:, n_pos_ex:, :]

        positive_prototype = positive_examples.mean(1)
        negative_prototype = negative_examples.mean(1)

        return positive_prototype, negative_prototype

    def reset_parameters(self):
        pass


class AttentionPrototyper(nn.Module):
    def __init__(self, d_model, *args, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.pos_pool = broccoli.vit.SequencePool(d_model)
        self.neg_pool = broccoli.vit.SequencePool(d_model)

    def forward(self, samples, labels=None):
        n_pos_ex = samples.size(1) // 2

        positive_examples = samples[:, :n_pos_ex, :]
        negative_examples = samples[:, n_pos_ex:, :]

        positive_prototype = self.pos_pool(positive_examples)
        negative_prototype = self.neg_pool(negative_examples)

        return positive_prototype, negative_prototype

    def reset_parameters(self):
        self.pos_pool.reset_parameters()
        self.neg_pool.reset_parameters()


class SenderGRULM(nn.Module):
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
        self.tau = kwargs["tau"]  # Gumbel-softmax tau, as in jayelm
        self.uniform_weight = kwargs["uniform_weight"]
        self.dropout = kwargs["dropout"]
        self.layers = kwargs["layers"]
        self.bidirectional = kwargs["bidirectional"]
        self.directions = 2 if self.bidirectional else 1

        # A constant, resolved once: nothing about it is learned or calibrated,
        #     so there is no buffer here and nothing enters the `state_dict`.
        self.logit_scale = logit_scale(
            kwargs["logit_scale_coefficient"], self.vocabulary
        )

        # Not state: per-batch diagnostics, read by `train.py` for metrics.csv.
        #     `logit_spread` is the standard deviation of the emittable logits
        #     *before* normalisation, and exists to disambiguate the two ways
        #     `realised_survival` can fall: the speaker learning a flatter
        #     policy, which is a finding, or its logit scale collapsing towards
        #     the LayerNorm epsilon, which is a fault. Both look identical in
        #     `realised_survival` alone.
        self.realised_survival = float("nan")
        self.logit_spread = float("nan")

        self.gru = nn.GRU(
            self.token_embedding_size,
            self.d_model,
            num_layers=self.layers,
            bias=True,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )

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

        Unlike `SenderTransformerLM`, whose embeddings are a function of the
            prototypes alone, this speaker is autoregressive: each embedding
            depends on the symbols sampled before it, so embeddings and
            message only correspond when they come from the *same* call.
            `forward` discards the embeddings, so any analysis needing the two
            to be paired must call this directly (as `Sender.speak` does).

        Returns:
            lang_tensor: (batch, message_length, vocabulary + 4)
            symbol_embeddings: (batch, message_length - 2, d_model *
                directions), one dense contextual vector per content symbol,
                taken before the vocabulary projection and sampling, or None
                when `return_states` is False. Stacking them is a copy the
                training path has no use for, so `forward` opts out.
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
        # Pre-gain logits for every position, kept so the exploration gain can
        #     be recalibrated once per batch rather than once per position.
        survival_logits = []
        spread_steps = []

        # Create and add SOS token
        sos_onehot = torch.zeros(
            batch_size,
            1,
            self.vocabulary + 4, # +4 for PAD, SOS, EOS, UNK
            device=device
        )  # Shape: (B, 1, V)
        sos_onehot[:, 0, data.language.SOS_IDX] = 1.0
        lang.append(sos_onehot)

        gru_in = sos_onehot @ self.token_embedding.weight  # Shape: (B, 1, D)

        # Main sampling loop (fixed length of message_length - 2)
        for i in range(self.message_length - 2):

            gru_out, states = self.gru(gru_in, states)

            step_embedding = gru_out[:, -1, :] # Shape: (B, D * Dir)
            symbol_embeddings.append(step_embedding)

            logits = self.outputs2vocab(step_embedding) # Shape: (B, V)

            if self.training:
                # Before normalisation, which is the whole point of it: this is
                #     the scale the normaliser has to work with.
                spread_steps.append(
                    logits[..., 4:].detach().float().std(-1).mean()
                )

            # This must come before the scale and the uniform weight mixing: it
            #     is what fixes the magnitude the scale is expressed against,
            #     and it would otherwise mess up the mixture.
            logits = layer_norm_logits(logits, self.vocabulary)

            # Masking comes first so that the uniform mixture below is spread
            #     over the emittable tokens only.
            logits = mask_reserved_tokens(logits)

            # Exploration is a training-time device only, so that the eval
            #     passes measure the learned policy rather than a deliberately
            #     noised one. Mirrors jayelm's emergent-generalization, which
            #     zeroes `uniform_weight` whenever the split is not `train`.
            if self.training:
                survival_logits.append(logits.detach())

                # Scale first, mixture second. The scale sets how much of the
                #     fixed 1.283-sd Gumbel noise the logits stand up to.
                #     Scaling *after* the mixture would undo the bounds the
                #     mixture exists to impose.
                logits = logits * self.logit_scale

                if self.uniform_weight > 0.0:
                    logits = flatten_logit_distribution(logits, self.uniform_weight)

                # 5. Gumbel-Softmax (hard=True)
                # This handles `argmax(logits + noise)` + straight-through gradient.
                # Note `tau` rescales the *soft* sample only: the hard forward sample
                #     is an argmax and so is invariant to it. It is a gradient knob,
                #     not an exploration knob — `logit_scale` is the latter, and
                #     `uniform_weight` puts a floor under it. Because the scale is
                #     a constant, the ratio between the two is fixed for the whole
                #     run, so the estimator sits at one operating point throughout.
                predicted_onehot = F.gumbel_softmax(
                    logits,
                    tau=self.tau,
                    hard=True,
                    dim=-1
                )
            else:
                # Greedy autoregressive decoding: eval measures the policy, so
                #     no noise, no mixture, no scale. The reserved tokens are
                #     -inf, so the argmax can never select one.
                predicted_onehot = F.one_hot(
                    logits.argmax(-1), self.vocabulary + 4
                ).to(logits.dtype)

            # 6. Prepare next input
            lang.append(predicted_onehot.unsqueeze(1))
            gru_in = (predicted_onehot.unsqueeze(1)) @ self.token_embedding.weight # (B, 1, D)

        # One measurement per batch, after sampling, on the pooled logits of
        #     every position. Doing it per position instead would read each
        #     position's statistics alone.
        if self.training:
            self.realised_survival = mean_winning_probability(
                torch.stack(survival_logits, 1).float(),
                self.logit_scale,
                self.uniform_weight,
            ).item()
            self.logit_spread = torch.stack(spread_steps).mean().item()

        # Add final EOS token
        eos_onehot = torch.zeros(batch_size, 1, self.vocabulary + 4, device=device)
        eos_onehot[:, 0, data.language.EOS_IDX] = 1.0
        lang.append(eos_onehot)

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
        We don't include options for greedy or epsilon-greedy generation as
            the former is only used in the parts of the code that relate to
            ACRe and the latter are by default not used (and are not
            commented upon in the original paper).
        """
        return self.decode(prototypes, return_states=False)[0]

    def reset_parameters(self):
        self.init_h.reset_parameters()
        self.gru.reset_parameters()
        self.outputs2vocab.reset_parameters()
        self.token_embedding.reset_parameters()
        self.realised_survival = float("nan")
        self.logit_spread = float("nan")


class SenderTransformerLM(nn.Module):
    def __init__(
        self,
        referent_embedding_size,
        **kwargs
    ):
        """
        ...
        
        https://arxiv.org/abs/2502.20604
        """
        super().__init__()
        self.referent_embedding_size = referent_embedding_size
        self.token_embedding_size = kwargs["token_embedding_size"]
        self.d_model = kwargs["d_model"]
        self.vocabulary = kwargs["vocabulary"]
        self.message_length = kwargs["message_length"]
        self.tau = kwargs["tau"]  # Gumbel-softmax tau, as in jayelm
        self.uniform_weight = kwargs["uniform_weight"]
        self.dropout = kwargs["dropout"]
        self.layers = kwargs["layers"]
        self.bidirectional = kwargs["bidirectional"]
        self.heads = kwargs["heads"]
        self.utility_tokens = kwargs["utility_tokens"]
        self.ff_ratio = kwargs["ff_ratio"]
        self.stochastic_depth = kwargs["stochastic_depth"]
        self.positional_heads = kwargs["positional_heads"]
        self.activation = model_util.get_activation(kwargs["activation"])
        self.absolute_position_embedding = kwargs["absolute_position_embedding"]
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
        self.alpha = kwargs["alpha"]
        self.beta = kwargs["beta"]

        # A constant, resolved once: nothing about it is learned or calibrated,
        #     so there is no buffer here and nothing enters the `state_dict`.
        self.logit_scale = logit_scale(
            kwargs["logit_scale_coefficient"], self.vocabulary
        )

        # Not state: per-batch diagnostics, read by `train.py` for metrics.csv.
        #     `logit_spread` is the standard deviation of the emittable logits
        #     *before* normalisation, and exists to disambiguate the two ways
        #     `realised_survival` can fall: the speaker learning a flatter
        #     policy, which is a finding, or its logit scale collapsing towards
        #     the LayerNorm epsilon, which is a fault. Both look identical in
        #     `realised_survival` alone.
        self.realised_survival = float("nan")
        self.logit_spread = float("nan")

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

        self.query = nn.Parameter(
            torch.empty(self.content_length, self.d_model)
        )

        self.query_layer_norm = nn.LayerNorm(self.d_model)
        self.referent_layer_norm = nn.LayerNorm(self.d_model)

        # As in `receiver.py`, every broccoli argument is set explicitly, including
        #     the inert ones, because broccoli's defaults have changed underneath
        #     this repository before. See the note at the top of that module.
        self.cross_attention = broccoli.transformer.MHAttention(
            self.d_model,
            self.heads,
            # Attention internals get their own setting, never the speaker's
            #     `dropout`, which regularises the prototypes going in. This
            #     used to read `self.dropout` while the listener's matching
            #     cross-attention took a separate constant, so raising the
            #     speaker's input regularisation silently rewired its attention
            #     and the two agents were regularised on different terms.
            dropout=self.cross_attention_dropout,
            causal=False, # Whole image informs whole initial message
            seq_len=self.content_length,
            linear_module=nn.Linear,
            bos_tokens=0,
            knocking_heads=False,
            # No positional information here: the query carries its own order
            #     and the prototypes are an unordered pair. `positional_heads`
            #     is inert while `rotary_embedding` is None, but is pinned
            #     anyway — broccoli defaults it to 0.25 on `MHAttention` and
            #     0.5 on `TransformerEncoder`, so the two are not the same.
            rotary_embedding=None,
            positional_heads=0.25,
            source_size=None,
            scaling="d",
        )

        self.transformer = broccoli.transformer.TransformerEncoder(
            self.content_length,
            self.d_model,
            self.layers,
            self.heads,
            absolute_position_embedding=self.absolute_position_embedding,
            relative_position_embedding=self.relative_position_embedding,
            positional_heads=self.positional_heads,
            # Derived, not configured: the sequence this block runs over is the
            #     message minus its SOS and EOS tokens.
            source_size=(self.content_length,),
            ff_ratio=self.ff_ratio,
            ff_inner_size=None, # inert: `ff_ratio` sizes the block instead
            activation=self.activation,
            activation_kwargs=None,
            ff_linear_module_up=None,
            ff_linear_module_down=None,
            # Not configurable, and pinned rather than promoted: broccoli's
            # `FeedforwardBlock` uses this only as a fallback --
            # `inner_dropout if inner_dropout is not None else dropout` -- and
            # `TransformerEncoder` always forwards `ff_inner_dropout` and
            # `ff_outer_dropout`, which default to 0.0 rather than None. So this
            # argument can never take effect, and TOML has no way to write the
            # None that would let it. Use the inner/outer knobs instead.
            ff_dropout=0.0,
            ff_inner_dropout=self.ff_inner_dropout,
            ff_outer_dropout=self.ff_outer_dropout,
            msa_dropout=self.self_attention_dropout,
            stochastic_depth=self.stochastic_depth,
            depthwise_linear_stochastic_depth=self.depthwise_linear_stochastic_depth,
            causal = not self.bidirectional,
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

        self.outputs2vocab = nn.Linear(
            self.d_model,
            self.vocabulary + 4 # +4 for PAD, SOS, EOS, UNK
        )

        self.reset_parameters()

    def embeddings(
        self,
        prototypes
    ):
        batch_size = prototypes[0].size(0)

        stack_prototypes = torch.stack(prototypes, 1) # To sequence

        normed_prototypes = self.referent_layer_norm(stack_prototypes)

        query = self.query.unsqueeze(0).expand(
            batch_size,
            self.content_length,
            self.d_model
        )

        normed_query = self.query_layer_norm(query)

        initial_sequence = self.cross_attention(
            normed_query, normed_prototypes, normed_prototypes
        ) # (batch, self.content_length, self.d_model)

        return self.transformer(initial_sequence)

    def decode(
        self,
        prototypes
    ):
        """
        Produce a message and the symbol embeddings that produced it, in a
            single pass. Mirrors `SenderGRULM.decode`.

        This speaker is not autoregressive, so unlike the GRU the embeddings
            here do not depend on which symbols were sampled. This method
            exists so that callers can treat the two speakers identically.

        Returns:
            onehot: (batch, message_length, vocabulary + 4)
            symbol_embeddings: (batch, message_length - 2, d_model), one
                dense contextual vector per content symbol, taken before the
                vocabulary projection and sampling
        """
        batch_size = prototypes[0].size(0)
        device = prototypes[0].device

        symbol_embeddings = self.embeddings(prototypes)

        logits = self.outputs2vocab(symbol_embeddings)

        if self.training:
            # Before normalisation — as in `SenderGRULM.decode`, see there.
            self.logit_spread = (
                logits[..., 4:].detach().float().std(-1).mean().item()
            )

        # This must come before the scale and the uniform weight mixing — as in
        #     `SenderGRULM.decode`, see the notes there.
        logits = layer_norm_logits(logits, self.vocabulary)

        # Mask first, then explore, training-time only — as in
        #     `SenderGRULM.decode`, see the notes there.
        logits = mask_reserved_tokens(logits)

        if self.training:
            survival_logits = logits.detach()

            # Scale first, mixture second; see `SenderGRULM.decode`.
            logits = logits * self.logit_scale

            if self.uniform_weight > 0.0:
                logits = flatten_logit_distribution(logits, self.uniform_weight)

            onehot_content = F.gumbel_softmax(
                logits,
                tau=self.tau,
                hard=True,
                dim=-1
            )

            # This speaker emits every position in one shot, so its logits are
            #     already pooled over positions.
            self.realised_survival = mean_winning_probability(
                survival_logits.float(), self.logit_scale, self.uniform_weight
            ).item()
        else:
            # Greedy: eval measures the policy. The reserved tokens are -inf,
            #     so the argmax can never select one.
            onehot_content = F.one_hot(
                logits.argmax(-1), self.vocabulary + 4
            ).to(logits.dtype)

        sos_onehot = torch.zeros(batch_size, 1, self.vocabulary + 4, device=device)
        sos_onehot[:, 0, data.language.SOS_IDX] = 1.0
        eos_onehot = torch.zeros(batch_size, 1, self.vocabulary + 4, device=device)
        eos_onehot[:, 0, data.language.EOS_IDX] = 1.0

        onehot = torch.cat([sos_onehot, onehot_content, eos_onehot], dim=1)

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
        self.cross_attention.reset_parameters()
        self.transformer.reset_parameters()
        self.outputs2vocab.reset_parameters()
        self.realised_survival = float("nan")
        self.logit_spread = float("nan")


class Sender(nn.Module):
    def __init__(
        self,
        feat_model: nn.Module,
        prototyper: nn.Module,
        language_model: nn.Module,
        vision_dropout: float= 0.5,
        prototype_dropout: float= 0.5
    ):
        """
        An agent that will receive one or more positive examples of a concept and
            one or more negative examples of a concept and will produce an utterance
            intended to communicate the concept

        Args:
            feat_model: The module used to produce embeddings from referents
            prototyper: The module used to create prototypes from positive and
                negative examples of referents
            language_model: The module used to create utterances based on prototypes
            vision_dropout: Dropout probability between the `feat_model` and the
                `prototyper`, i.e. on per-image embeddings, before pooling
            prototype_dropout: Dropout probability between the `prototyper` and
                the `language_model`, i.e. on the pooled concept vectors. This
                is where jayelm's single `--dropout` sits (on the speaker side);
                dropping features before the pool is much weaker, since the
                average over n/2 examples largely restores them.
        """
        super().__init__()
        self.feat_model = feat_model
        self.feat_size = feat_model.final_feat_dim
        self.prototyper = prototyper
        self.language_model = language_model
        self.vision_dropout = nn.Dropout(p=vision_dropout)
        self.prototype_dropout = nn.Dropout(p=prototype_dropout)

    def embed_images(self, samples):
        """
        Embed all the referent images in a batch.
        Input size is (batch, referents, image dim 1, image dim 2, etc.)
        We reshape to (batch_size * n_obj, *rest) to get a batch of images, then
            send them through the computer vision model, then reshape back to the
            shape needed for the task.
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

        prototypes = self.prototyper(self.embed_images(samples), targets)

        # Applied per prototype rather than to the concatenation, which is the
        #     same thing: dropout masks each element independently.
        return tuple(self.prototype_dropout(p) for p in prototypes)

    def speak(self, samples, targets):
        """
        Produce a message, the symbol embeddings behind it, and the concepts
            that prompted it, all from a single pass.

        This is what compositionality analysis needs: the soft signal
            distances compare symbol embeddings between symbols that were
            actually emitted, and semantic distance is measured between the
            concepts, so all three must come from the same forward pass.
            Fetching them through separate calls would resample both the
            vision dropout mask and (for the GRU speaker) the message itself.

        Returns:
            messages: (batch, message_length, vocabulary + 4)
            symbol_embeddings: (batch, message_length - 2, embedding size),
                one dense contextual vector per content symbol, positionally
                aligned with the content symbols of `messages`, i.e. with the
                output of `trim_messages`
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
        #     the vision model under a fresh `vision_dropout` mask, so the
        #     returned concepts would not be the ones that produced the message.
        prototypes = self.get_prototypes(samples, targets)

        messages = self.language_model(
            prototypes,
            **kwargs
        )

        return messages, torch.cat(prototypes, 1)

    def reset_parameters(self):
        # No `hasattr` guard, matching `Receiver.reset_parameters`. The guard
        #     existed for `ViT2`, which had no `reset_parameters`; it now does,
        #     and every other feature model already did. A guard here turns a
        #     missing method into a silently skipped backbone rather than an
        #     error, which is how the speaker's ViT went unreset.
        self.feat_model.reset_parameters()
        self.prototyper.reset_parameters()
        self.language_model.reset_parameters()