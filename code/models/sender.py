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

    Args:
        logits: (..., vocabulary + 4), reserved tokens first
        vocabulary: number of emittable tokens

    Returns:
        A tensor of the same shape, with the emittable slice normalised
    """
    return torch.cat(
        [logits[..., :4], F.layer_norm(logits[..., 4:], (vocabulary,))],
        dim=-1,
    )


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


# The exploration gain is an EMA over per-batch solves, smoothed in log space
#     because gains are multiplicative. The clamp is a backstop only: a run that
#     pins to either end has something wrong upstream of the channel.
#
# The momentum is `max(1 / (t + 1), EXPLORATION_GAIN_MOMENTUM)`: a cumulative
#     average that decays into a fixed-rate EMA once the floor bites, as
#     `nn.BatchNorm1d` does with `momentum=None`. Early on there is no history
#     worth keeping and the gain is moving fastest, so new solves should
#     dominate; later the floor holds the response time constant.
#
# The floor is deliberately fast. Smoothing trades lag against noise, and there
#     is very little noise here to buy: the median step-to-step change in the
#     solved gain is ~0.1% in log terms, against drift of an order of magnitude
#     over a few dozen batches early in training. On a recorded trajectory,
#     dropping the floor from 0.01 to 0.1 halved the mean deviation of realised
#     survival from its target (0.086 -> 0.042), while the cumulative-average
#     warm-up on its own contributed ~0.002. At ~650 updates per ShapeWorld
#     epoch a floor of 0.1 is a time constant of well under a hundredth of an
#     epoch, so the gain tracks continuously across a 100-epoch run.
EXPLORATION_GAIN_MOMENTUM = 0.1
EXPLORATION_GAIN_MIN = 1e-2
EXPLORATION_GAIN_MAX = 1e4


def mean_winning_probability(
    logits: torch.Tensor,
    gain: float,
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

    Applies the real sampling pipeline in the real order — gain first, then the
        uniform mixture — so that the mixture's bounds hold. Mixing first and
        scaling afterwards destroys them.

    Args:
        logits: (..., vocabulary + 4), reserved tokens already masked to -inf
        gain: the multiplier applied before mixing
        uniform_weight: as in `flatten_logit_distribution`

    Returns:
        A scalar tensor, the mean over all slots of the winning token's
            post-mixing probability
    """
    scaled = logits * gain
    if uniform_weight > 0.0:
        scaled = flatten_logit_distribution(scaled, uniform_weight)
    return scaled.softmax(-1).max(-1).values.mean()


def calibrate_exploration_gain(
    logits: torch.Tensor,
    token_exploration_rate: float,
    uniform_weight: float,
    iterations: int = 25,
    lo: float = EXPLORATION_GAIN_MIN,
    hi: float = EXPLORATION_GAIN_MAX,
) -> torch.Tensor:
    """
    Solve for the logit gain at which the expected fraction of symbols flipped
        by the Gumbel noise equals `token_exploration_rate`.

    This is what turns exploration from an accident of architecture into a
        stated number. The Gumbel noise has a fixed standard deviation of 1.283,
        so channel fidelity is set entirely by the scale of the speaker's
        logits — which varied by two orders of magnitude across an
        architecture ladder, from a channel that passed 99% of symbols to one
        that passed 24%. Normalising the logits fixes their magnitude; this
        fixes what that magnitude buys.

    Why a runtime solve rather than a closed form: survival depends on the
        *shape* of each logit vector, not only on its scale, so even
        normalised speakers realise different rates at a common gain. Why
        bisection rather than learning it: the largest move of any
        gradient-updated 1-D speaker parameter over 90 epochs was 0.12, against
        an AdamW ceiling of 1.49, so a learnable gain cannot travel far enough
        to matter. Why not `tau`: it divides *after* the noise, so the hard
        sample is invariant to it — it is a gradient-estimator knob only.

    Mean winning probability is monotone increasing in the gain, so plain
        bisection on `log(gain)` is valid. Each trial runs the real pipeline via
        `mean_winning_probability`, which calls `flatten_logit_distribution`
        itself, so the calibration and the sampler cannot drift apart.

    Args:
        logits: detached, normalised, reserved-masked logits, (..., vocabulary + 4)
        token_exploration_rate: target fraction of symbols flipped by the noise
        uniform_weight: as in `flatten_logit_distribution`
        iterations: bisection steps
        lo, hi: bracket for the gain

    Returns:
        A scalar tensor holding the solved gain
    """
    target = 1.0 - token_exploration_rate

    lo_log, hi_log = math.log(lo), math.log(hi)
    for _ in range(iterations):
        mid_log = 0.5 * (lo_log + hi_log)
        realised = mean_winning_probability(
            logits, math.exp(mid_log), uniform_weight
        )
        if realised < target:
            lo_log = mid_log
        else:
            hi_log = mid_log

    return torch.as_tensor(
        math.exp(0.5 * (lo_log + hi_log)),
        dtype=logits.dtype,
        device=logits.device,
    )


def exploration_rate_floor(uniform_weight: float, vocabulary: int) -> float:
    """
    The smallest token exploration rate the uniform mixture can realise.

    Mixing caps a slot's winner at `1 - w + w/V` however sharp the underlying
        logits are, so at least `w * (1 - 1/V)` of symbols are flipped no matter
        what the gain does. That is a permanent corruption rate, and it is
        intended: irreducible late-training exploration is the point.
    """
    return uniform_weight * (1.0 - 1.0 / vocabulary)


def check_exploration_rate_floor(
    token_exploration_rate: float,
    uniform_weight: float,
    vocabulary: int,
) -> None:
    """
    Warn when the requested exploration rate is below what the uniform mixture
        allows, in which case the calibration will pin at the bracket's top and
        the realised rate will sit at the floor rather than at the request.
    """
    floor = exploration_rate_floor(uniform_weight, vocabulary)
    if token_exploration_rate < floor:
        warnings.warn(
            f"`token_exploration_rate` of {token_exploration_rate} is below "
            f"the floor of {floor:.4f} imposed by `uniform_weight` "
            f"({uniform_weight}) at a vocabulary of {vocabulary}: uniform "
            f"mixing caps a slot's winning probability at 1 - w + w/V, so at "
            f"least w * (1 - 1/V) of symbols are flipped whatever the gain is. "
            f"The request is unreachable and the realised rate will sit at the "
            f"floor. Lower `uniform_weight` to go below it."
        )


@torch.no_grad()
def update_exploration_gain(speaker: nn.Module, logits: torch.Tensor) -> None:
    """
    Recalibrate a speaker's `exploration_gain` buffer from one batch of logits.

    Called once per decode, after sampling, on the train pass only. The gain
        used for *this* decode is therefore the buffer's value from before the
        update, which is the only option for the autoregressive speaker: it
        cannot know its later positions' logits before sampling the earlier
        ones.

    Smoothed in log space, since gains compose multiplicatively, at a momentum
        of `max(1 / (t + 1), EXPLORATION_GAIN_MOMENTUM)` over the number of
        updates so far. The first update therefore takes the solve outright
        (1/1 = 1) rather than needing a special case, so batch one is not
        sampled at a gain of 1.0; the next few still weight new solves heavily,
        which is when the gain moves fastest; and the floor takes over once
        there is enough history to average. See the note on the constant for
        why the floor is set where it is.

    Because `exploration_gain_updates` is a buffer it survives checkpointing, so
        a resumed run continues at its established momentum instead of dropping
        back into fast adaptation.

    If the `exploration_gain` trace in metrics.csv looks jittery, lower
        `EXPLORATION_GAIN_MOMENTUM` rather than raising the bisection count; if
        `realised_survival` sits away from `token_exploration_rate`, raise it.

    Args:
        speaker: a speaker carrying `exploration_gain`, `exploration_gain_updates`,
            `token_exploration_rate` and `uniform_weight`
        logits: normalised, reserved-masked logits for every position of the
            batch, (..., vocabulary + 4). Detached and promoted to float32 here,
            since `train.py` runs the forward pass under autocast.
    """
    logits = logits.detach().float()

    solved = calibrate_exploration_gain(
        logits,
        speaker.token_exploration_rate,
        speaker.uniform_weight,
    )

    momentum = max(
        1.0 / (int(speaker.exploration_gain_updates) + 1),
        EXPLORATION_GAIN_MOMENTUM,
    )
    gain = torch.exp(
        (1.0 - momentum) * torch.log(speaker.exploration_gain.float())
        + momentum * torch.log(solved)
    )

    gain = gain.clamp(EXPLORATION_GAIN_MIN, EXPLORATION_GAIN_MAX)

    speaker.exploration_gain.copy_(gain)
    speaker.exploration_gain_updates += 1

    # Reported per epoch as `realised_survival`, and read at the gain the buffer
    #     actually holds rather than at the freshly solved one, so that it
    #     confirms the EMA has converged rather than restating the target.
    speaker.realised_survival = mean_winning_probability(
        logits, speaker.exploration_gain.item(), speaker.uniform_weight
    ).item()


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
        self.token_exploration_rate = kwargs["token_exploration_rate"]
        self.uniform_weight = kwargs["uniform_weight"]
        self.layer_norm_logits = kwargs["layer_norm_logits"]
        self.dropout = kwargs["dropout"]
        self.layers = kwargs["layers"]
        self.bidirectional = kwargs["bidirectional"]
        self.directions = 2 if self.bidirectional else 1

        check_exploration_rate_floor(
            self.token_exploration_rate, self.uniform_weight, self.vocabulary
        )

        # Buffers, not parameters: the gain is set by calibration, never by
        #     gradient, and it has to survive checkpoint and resume.
        self.register_buffer("exploration_gain", torch.tensor(1.0))
        self.register_buffer(
            "exploration_gain_updates", torch.tensor(0, dtype=torch.long)
        )
        # Not state: a per-batch diagnostic, read by `train.py` for metrics.csv.
        self.realised_survival = float("nan")

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
        calibration_logits = []

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

            if self.layer_norm_logits:
                # This must come before the gain and the uniform weight mixing:
                #     it is what fixes the magnitude the gain is calibrated
                #     against, and it would otherwise mess up the mixture.
                logits = layer_norm_logits(logits, self.vocabulary)

            # Masking comes first so that the uniform mixture below is spread
            #     over the emittable tokens only.
            logits = mask_reserved_tokens(logits)

            # Exploration is a training-time device only, so that the eval
            #     passes measure the learned policy rather than a deliberately
            #     noised one. Mirrors jayelm's emergent-generalization, which
            #     zeroes `uniform_weight` whenever the split is not `train`.
            if self.training:
                calibration_logits.append(logits.detach())

                # Gain first, mixture second. The gain sets how much of the
                #     fixed 1.283-sd Gumbel noise the logits stand up to, and
                #     is calibrated below to deliver
                #     `token_exploration_rate`. Scaling *after* the mixture
                #     would undo the bounds the mixture exists to impose.
                #
                # Cloned because `update_exploration_gain` writes the buffer in
                #     place at the end of the decode, and autograd rejects a
                #     saved tensor that has been mutated since. Cloning keeps
                #     that on the device, where `.item()` would sync.
                logits = logits * self.exploration_gain.clone()

                if self.uniform_weight > 0.0:
                    logits = flatten_logit_distribution(logits, self.uniform_weight)

                # 5. Gumbel-Softmax (hard=True)
                # This handles `argmax(logits + noise)` + straight-through gradient.
                # Note `tau` rescales the *soft* sample only: the hard forward sample
                #     is an argmax and so is invariant to it. It is a gradient knob,
                #     not an exploration knob — `exploration_gain` is the latter,
                #     and `uniform_weight` puts a floor under it.
                predicted_onehot = F.gumbel_softmax(
                    logits,
                    tau=self.tau,
                    hard=True,
                    dim=-1
                )
            else:
                # Greedy autoregressive decoding: eval measures the policy, so
                #     no noise, no mixture, no gain. The reserved tokens are
                #     -inf, so the argmax can never select one.
                predicted_onehot = F.one_hot(
                    logits.argmax(-1), self.vocabulary + 4
                ).to(logits.dtype)

            # 6. Prepare next input
            lang.append(predicted_onehot.unsqueeze(1))
            gru_in = (predicted_onehot.unsqueeze(1)) @ self.token_embedding.weight # (B, 1, D)

        # One recalibration per batch, after sampling, on the pooled logits of
        #     every position. Doing it per position instead would recalibrate
        #     five times a batch and read each position's statistics alone.
        if self.training:
            update_exploration_gain(self, torch.stack(calibration_logits, 1))

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
        self.token_exploration_rate = kwargs["token_exploration_rate"]
        self.uniform_weight = kwargs["uniform_weight"]
        self.layer_norm_logits = kwargs["layer_norm_logits"]
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

        check_exploration_rate_floor(
            self.token_exploration_rate, self.uniform_weight, self.vocabulary
        )

        # Buffers, not parameters: the gain is set by calibration, never by
        #     gradient, and it has to survive checkpoint and resume.
        self.register_buffer("exploration_gain", torch.tensor(1.0))
        self.register_buffer(
            "exploration_gain_updates", torch.tensor(0, dtype=torch.long)
        )
        # Not state: a per-batch diagnostic, read by `train.py` for metrics.csv.
        self.realised_survival = float("nan")

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

        if self.layer_norm_logits:
            # This must come before the gain and the uniform weight mixing —
            #     as in `SenderGRULM.decode`, see the notes there.
            logits = layer_norm_logits(logits, self.vocabulary)

        # Mask first, then explore, training-time only — as in
        #     `SenderGRULM.decode`, see the notes there.
        logits = mask_reserved_tokens(logits)

        if self.training:
            calibration_logits = logits.detach()

            # Gain first, mixture second, and cloned so the in-place buffer
            #     update below does not invalidate it; see `SenderGRULM.decode`.
            logits = logits * self.exploration_gain.clone()

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
            update_exploration_gain(self, calibration_logits)
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
        if hasattr(self.feat_model, 'reset_parameters'):
            self.feat_model.reset_parameters()
        self.prototyper.reset_parameters()
        self.language_model.reset_parameters()