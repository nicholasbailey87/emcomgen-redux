"""
Listener models

The listener is two swappable slots, mirroring the speaker's
`prototyper` / `language_model` split:

    language_model  encodes the message           -> (batch, slots, width)
    discriminator   scores the candidates from it -> (batch, n_objects)

`Receiver` owns the composition, the token embedding, and every interface
between the vision backbone or the message encoder and a slot: each slot
declares the widths it wants and `Receiver` delivers each input at that width
in a stated distribution. See `model_util.LinearInterface` for the rule -- it
is the speaker's `adapter` too. Both slots are named in `[receiver]` and
configured from `[receiver_language_model]` and
`[receiver_discriminator]`: four language models by one discriminator.

                                         BilinearDiscriminator
    ReceiverGRULM                        the historical baseline
    ReceiverTransformerAutoregressiveLM  the GRU's reading regime, attention
    ReceiverTransformerBidirectionalLM   the whole message at once
    ReceiverCrossAttentionLM             reads the candidates too

The split exists so a rung moves one half at a time. Before it, one `comparer`
key chose both halves at once, so a rung that swapped the GRU comparer for the
cross-attention one changed the message encoder *and* the comparison, and
"attention helps" could not be attributed to either. See docs/architecture.md.

There used to be a second discriminator, an attention discriminator in which the
candidates read each other. It was removed on 2026-10-01: like
`ReceiverCrossAttentionLM` it can score "which cluster" without reading the
message, and `train_shuffled_message_acc` showed it doing so. See
docs/anecdotes.md.

**Exactly one message encoder, always.** No configuration builds two, and if
one ever looks necessary the slot contract is wrong.
"""

import math

import torch
import torch.nn as nn

import broccoli

from . import model_util
from . import transformer_decoder

# Re-exported under the name it has always had, and shared with the speaker
#     rather than restated: below the 1e-5 default the normaliser quietly stops
#     normalising and the score's magnitude goes back to the backbone. Two
#     modules writing `1e-12` independently is two places one of them can drift.
#     See `model_util.LAYER_NORM_EPS` and docs/channel.md.
LAYER_NORM_EPS = model_util.LAYER_NORM_EPS

# Every broccoli module below is constructed with its full argument list, even
#     where an argument is inert under the current settings, because broccoli's
#     defaults are not a stable interface. See docs/broccoli.md.


class ScoreVolume:
    """
    The listener's one degree of freedom over how loudly it states a conclusion.
        The counterpart of the speaker's `GumbelChannel.logit_scale`: both are
        lone learned scalars in front of a normalised quantity, both are plain
        products, and both take the same rate.
        The speaker's is bounded above by projection where this one is not --
        a volume has no natural ceiling, a channel scale does. See
        docs/architecture.md and docs/channel.md.

    A mixin rather than a submodule: `log_score_scale` stays registered on the
        discriminator itself, so the `state_dict` key is the one
        `split_out_parameter` matches by suffix and the one earlier checkpoints
        were written against.

    The readout is `score_scale * scores + score_bias`: a volume and an offset,
        in that order. The scalar still sits in front of a normalised quantity
        -- that pairing is what stops the volume meaning something different
        under every backbone -- but the normalising happens on the
        discriminator's *inputs* rather than on its output.
        Both operands of `BilinearDiscriminator`'s bilinear form arrive
        layer-normed -- from `Receiver`'s interfaces, unconditionally -- so its
        score opens at `1 / sqrt(3)` at any width and any backbone by
        construction, and there is nothing left for a normaliser downstream to
        fix.

    **Why there is an offset at all.** `train.py` decides on `lis_scores > 0`,
        so the threshold is a fixed origin and the listener has to place its
        scores against it. Without this the bilinear arm had no bias anywhere,
        `bilinear` being built `bias=False` and the readout a bare multiply.
        The bilinear score for candidate `j` is `LN(r_j) . proj`, so the only
        way to move all candidates together was for `proj` to align with
        whatever direction the candidates have in common, which is
        data-dependent and spends discriminative capacity in that direction.

    **Why downstream of the volume.** An offset applied before the scale is multiplied by it, so
        the threshold would slide every time the listener changed how loudly it
        spoke -- and `score_scale` moves fast, at `score_scale_lr`. Downstream,
        it is an offset on the score itself and the two parameters say
        independent things.

    **What the offset cannot do.** Games are balanced 10 positive / 10
        negative, so the loss-optimal *global* offset is near zero and
        `score_bias` should be expected to sit there. It corrects a systematic
        offset in where the scores sit; it cannot correct a per-game one, and
        the bilinear score's per-game mean is `mean_j(LN(r_j)) . proj`, which
        varies by game. If it moves and accuracy does not, the offset was
        per-game, no scalar reaches that, and the answer is a different readout
        rather than a bigger bias.

        It is also not what makes a run start. Rung 9's 2026-08-27 run sat at
        chance for ten epochs with `train_loss` at 0.6935 against `ln 2` =
        0.6931 -- the trivial optimum of scoring everything near zero, which is
        the point an offset gets you *to*. There was no headroom in it. That
        flat start was the tau coupling pinning the speaker's then-learned
        channel scale, which has since been removed along with the learned
        scale itself. See docs/channel.md.

    **Why not standardise the score.** `7b10d47` read out
        `score_scale * standardise(scores)`, dividing each game by the spread of
        its own candidate scores. Two reasons it is gone, and the weaker one is
        listed second deliberately.

        It is redundant. Both of `BilinearDiscriminator`'s operands arrive
        normalised, so the score is already backbone-independent and already
        opens at a stated number. A second normaliser downstream of that buys
        nothing and costs the calibration -- `7b10d47` deleted the exact
        `1/sqrt(d)` on the grounds that a standardise divides any constant out,
        which traded an analytic opening for an empirical one.

        It re-weights games by their own margin, but *only once the listener can
        already discriminate*, which is not the regime it was suspected in.
        Measured on this module: with `bilinear.weight` at its random init the
        score's spread is 0.567 whether the message carries the separating
        direction or pure noise, so standardise is a uniform 1.77x there and
        AdamW cancels it. With a listener that can read the message the spreads
        do come apart -- 4.20 against 0.98 -- and standardise then damps the
        informative games relative to the noise ones by about 1.4x. Real, wrong
        way round, and far too small to be the 382-445x slowdown in the sender's
        pre-channel parameters that the 2026-08-26 rung 9 and 10 runs showed.
        **That correlation is unexplained.** `standardise` is present in exactly
        the frozen runs and absent from every run that was merely dead, but this
        measurement rules out the obvious mechanism, so do not read the removal
        as a diagnosis. See docs/anecdotes.md.

    The volume is therefore shared with the weights again.
        `bilinear_weight_norm` means volume as well as direction, as it did
        before `7b10d47`: nothing downstream divides a rescaling of
        `bilinear.weight` back out.
    """

    def _init_score_volume(self, scale=True, bias=True):
        """
        Call from `__init__` where the parameters should be created: creation
            order fixes which RNG draw every later parameter gets.

        **Two gates rather than one**, because the volume and the offset answer
            two questions and the config asks them separately:
            `[receiver_discriminator] scale_score` builds `log_score_scale` and
            `bias_score` builds `score_bias`. Both default true, which is the
            arithmetic every recorded run has. A listener can now have a
            threshold without a loudness, or the reverse; under the single flag
            these keys replace it could only have both or neither, which was
            never a decision anybody made -- it was one boolean standing in for
            two.

        There is no third question here. The `1/sqrt(d)` calibration these
            gates used to travel with is unconditional and lives in
            `BilinearDiscriminator.forward`: it is what makes the score open at
            `1/sqrt(3)` under any width and any backbone, so it is design and
            not a hypothesis. See that method.

        Absent rather than frozen, so `split_out_parameter`'s suffix
            match and `SCALAR_GROUPS`' membership see the truth.
        """
        self.learns_score_scale = scale
        self.learns_score_bias = bias

        if self.learns_score_scale:
            # Stored as its log so `exp` keeps it strictly positive: gradient
            #     descent cannot walk a volume through zero and out the far
            #     side. Opens at 1.0, which on the bilinear arm leaves the score
            #     at its own calibrated opening of `1 / sqrt(3)` -- see
            #     `BilinearDiscriminator.forward`.
            self.log_score_scale = nn.Parameter(torch.zeros(()))

        if self.learns_score_bias:
            # Not a log, unlike the volume: an offset is signed, and zero is
            #     both where it opens and a value it must be able to return to.
            self.score_bias = nn.Parameter(torch.zeros(()))

    @property
    def score_scale(self):
        """
        The multiplier applied to the score, always positive. Read here rather
            than exponentiating at the use site so `forward` and the metrics
            column cannot drift apart.
        """
        return self.log_score_scale.exp()

    def readout(self, scores):
        """
        Apply the volume, then the offset. Nothing else -- the normalising this
            used to do is on the discriminator's inputs instead, where it cannot
            put the game's own margin in the denominator. See the class
            docstring, including why the offset is second.

        Each half is applied only if it exists, so this is the identity on a
            discriminator built with both gates off.

        The two are independent: `scale_score = false, bias_score = true` gives
            `scores + score_bias`, a threshold on an uncalibrated-loudness
            score, and the reverse gives a volume with the origin left where the
            calibration puts it. Both are reachable configurations rather than
            accidents of one flag.

        **The volume is a plain product**, so `d/dscores` is `score_scale`
            and a listener turning itself down scales down what reaches the
            speaker. Round seven's argument is why that is harmless: AdamW
            cancels a uniform factor on a parameter's gradient. Rounds six and
            nine hid the volume from the backward pass with
            `scale_without_attenuating`; round eleven removed that helper. See
            docs/anecdotes.md.
        """
        if self.learns_score_scale:
            scores = self.score_scale * scores

        if self.learns_score_bias:
            scores = scores + self.score_bias

        return scores

    def reset_score_volume(self):
        """
        Put the volume back to its 1.0 opening and the offset back to zero, so
            a reset does not leave a trained confidence or a trained threshold
            behind a fresh listener. Each is reset only if it was built.
        """
        with torch.no_grad():
            if self.learns_score_scale:
                self.log_score_scale.zero_()

            if self.learns_score_bias:
                self.score_bias.zero_()


# --------------------------------------------------------------------------
# Language models. Each takes `(messages, referents)` and returns
#     `(batch, slots, output_size)`.
# --------------------------------------------------------------------------

class ReceiverGRULM(nn.Module):
    def __init__(
        self,
        referent_embedding_size,
        **kwargs
    ):
        """
        Read the message with a GRU and return its final state.

        `referents` is accepted and ignored: this is an absolute encoding of
            the message, with no view of what it is being compared against. A
            uniform signature is what makes the slot swappable, and an unused
            argument is cheaper than dispatching on class at the call site.

        Because `referent_input_size` is `None`, `Receiver` builds no referent
            interface for this slot and passes `None` here. That the argument is
            ignored is therefore structural rather than a convention: there is
            no tensor to ignore.

        Returns a length-1 sequence rather than a bare vector so the slot
            contract is one shape for every language model.
            `BilinearDiscriminator` takes the last position, which is the
            identity here.

        Args:
            referent_embedding_size: recorded and not used. This slot reads no
                referents at all. Kept on the signature because it is the slot
                contract's shape.
        """
        super().__init__()
        self.referent_embedding_size = referent_embedding_size
        self.token_embedding_size = kwargs["token_embedding_size"]
        self.d_model = kwargs["d_model"]
        self.layers = kwargs["layers"]
        self.bidirectional = kwargs["bidirectional"]

        self.gru = nn.GRU(
            self.token_embedding_size,
            self.d_model,
            num_layers=self.layers,
            bias=True,
            batch_first=True,
            # Pinned at 0.0 to match jayelm, and inert at the default anyway:
            #     `nn.GRU` applies this *between* layers and `layers` is back to
            #     1, so it would only become a live decision under a config that
            #     deepened the stack. The listener's regularisation is
            #     `[receiver] dropout`, which masks the referents at the end
            #     of each of `Receiver`'s referent interfaces -- see
            #     docs/architecture.md.
            dropout=0.0,
            bidirectional=self.bidirectional
        )

    @property
    def referent_input_size(self):
        """
        `None`: this slot does not read the candidate set at all, so `Receiver`
            builds it no referent interface and hands it `None`. See
            `Receiver.__init__` for the declaration contract.
        """
        return None

    @property
    def message_input_size(self):
        """
        `None`, and for a different reason than `referent_input_size`'s: a
            language model is the thing that *produces* a message
            representation, so there is no message interface to declare. It
            reads the token embeddings directly.
        """
        return None

    @property
    def output_size(self):
        """
        The width the discriminator's message interface must adapt from. Read
            here rather than recomputed at the call site so the two cannot drift
            apart.
        """
        return self.d_model * 2 if self.bidirectional else self.d_model

    def forward(
        self,
        messages: torch.Tensor, # (batch, seq_len, token_embedding_size)
        referents: torch.Tensor, # ignored, and `None` from `Receiver`
        ) -> torch.Tensor: # -> (batch, 1, output_size)
        token_embeddings, _ = self.gru(messages) # (b, seq, directions * d_model)

        # Taking timestep -1 gives the state after the GRU has consumed the
        #     last *slot*, not the last real token. Correct only because our
        #     messages are never padded; see docs/dubious-claims.md for what
        #     would break it.
        if self.bidirectional:
            final_state_of_forward_pass = token_embeddings[:, -1, :self.d_model]
            final_state_of_backward_pass = token_embeddings[:, 0, self.d_model:]
            message_embeddings = torch.cat(
                (
                    final_state_of_forward_pass,
                    final_state_of_backward_pass
                ),
                dim=1
            )
        else:
            # Standard unidirectional extraction
            message_embeddings = token_embeddings[:, -1, ...]

        return message_embeddings.unsqueeze(1)

    def reset_parameters(self):
        self.gru.reset_parameters()


class ReceiverTransformerLM(nn.Module):
    def __init__(
        self,
        referent_embedding_size,
        **kwargs
    ):
        """
        Read the message with a Transformer encoder and pool it to one vector.
            The Transformer counterpart of `ReceiverGRULM`, in two arms selected
            by `bidirectional`: a causal stack (False) or an unmasked one (True).

        The shared implementation, following `SenderTransformerLM`. Configs
            select one of the two subclasses below, which set `bidirectional`
            themselves; `parse_config` refuses this name, whose one rate key
            would cover both arms.

        **It never sees the candidates.** Like the GRU, and unlike
            `ReceiverCrossAttentionLM`, this is an absolute encoding of the
            message: `referent_input_size` is `None`, so `Receiver` builds no
            referent interface for it and passes `None` in. That is the point of
            the class. A message encoder that cross-attends into the candidate
            set gets a summary of the set whatever the tokens say, and scoring
            against it is the concept game's clustering shortcut --
            `lr_sweep_6_receiver_cross_attention_lm` measured
            `train_shuffled_message_acc` at ~0.65 against live ~0.75 on every
            arm. See docs/architecture.md.

        Readout is a `broccoli.vit.SequencePool` on both arms, returned as a
            length-1 sequence, so `BilinearDiscriminator`'s last-slot read is the
            identity here as it is for the GRU. The causal arm's pool reads every
            position, each of which has seen only its prefix -- the pool is
            where the whole message is first combined, which is what a causal
            reading regime under a learned readout amounts to.

        Args:
            referent_embedding_size: recorded and not used. This slot reads no
                referents at all. Kept on the signature because it is the slot
                contract's shape.
        """
        super().__init__()
        self.referent_embedding_size = referent_embedding_size
        self.token_embedding_size = kwargs["token_embedding_size"]
        self.d_model = kwargs["d_model"]
        self.message_length = kwargs["message_length"]
        self.layers = kwargs["layers"]
        self.heads = kwargs["heads"]
        self.utility_tokens = kwargs["utility_tokens"]
        self.bidirectional = kwargs["bidirectional"]
        self.ff_inner_size = kwargs["ff_inner_size"]
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

        # The encoder form: two sublayers to the block, self-attention and a
        #     feedforward, and nothing to cross-attend into. See docs/broccoli.md.
        self.alpha, self.beta = model_util.resolve_residual_scaling(
            kwargs["alpha"], kwargs["beta"], self.layers, decoder=False,
        )

        # Suppressed unless the stack is deep enough for a depth ramp to mean
        #     anything, as on `ReceiverCrossAttentionLM`.
        self.stochastic_depth = (
            kwargs["stochastic_depth"] if self.layers > 1 else 0.0
        )

        # Reads the token embeddings, this slot's raw input; not an interface
        #     in `Receiver`'s sense. See `ReceiverCrossAttentionLM`.
        self.message_adapter = nn.Linear(
            self.token_embedding_size,
            self.d_model
        )

        self.encoder = broccoli.transformer.TransformerEncoder(
            self.message_length,
            self.d_model,
            self.layers,
            self.heads,
            # Pinned False, and no longer a config option; every stack here
            #     runs rotary. See docs/broccoli.md.
            absolute_position_embedding=False,
            relative_position_embedding=self.relative_position_embedding,
            # Pinned at 1.0, not configurable -- see `ViT2` for the argument.
            positional_heads=1.0,
            # Derived from the data: this stack reads the message.
            source_size=(self.message_length,),
            # `ff_ratio` None so that `ff_inner_size` is the live knob; note
            #     broccoli's `ViT` resolves the two the other way round.
            ff_ratio=None,
            ff_inner_size=self.ff_inner_size,
            activation=self.activation,
            activation_kwargs=None,
            ff_linear_module_up=None,
            ff_linear_module_down=None,
            # Pinned rather than promoted: this argument can never take effect.
            #     Use the inner/outer knobs instead. See docs/broccoli.md.
            ff_dropout=0.0,
            ff_inner_dropout=self.ff_inner_dropout,
            ff_outer_dropout=self.ff_outer_dropout,
            msa_dropout=self.self_attention_dropout,
            stochastic_depth=self.stochastic_depth,
            depthwise_linear_stochastic_depth=self.depthwise_linear_stochastic_depth,
            # The whole of what the two arms differ by.
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

        self.pool = broccoli.vit.SequencePool(self.d_model)

    @property
    def referent_input_size(self):
        """`None`: this slot never reads the candidates. See the class docstring."""
        return None

    @property
    def message_input_size(self):
        """`None`: this slot is the message encoder. See `ReceiverGRULM`."""
        return None

    @property
    def output_size(self):
        return self.d_model

    def encode(
        self,
        messages: torch.Tensor, # (batch, seq_len, token_embedding_size)
        ) -> torch.Tensor: # -> (batch, seq_len, d_model)
        """
        The per-position outputs, before the pool. Separate from `forward` so a
            test can check the causal arm's mask where it acts.
        """
        return self.encoder(self.message_adapter(messages))

    def forward(
        self,
        messages: torch.Tensor, # (batch, seq_len, token_embedding_size)
        referents: torch.Tensor, # ignored, and `None` from `Receiver`
        ) -> torch.Tensor: # -> (batch, 1, d_model)
        return self.pool(self.encode(messages)).unsqueeze(1)

    def reset_parameters(self):
        self.message_adapter.reset_parameters()
        self.encoder.reset_parameters()
        self.pool.reset_parameters()


class ReceiverTransformerAutoregressiveLM(ReceiverTransformerLM):
    def __init__(self, referent_embedding_size, **kwargs):
        """
        The causal arm of `ReceiverTransformerLM`: each position reads only the
            symbols before it, as the GRU does. Any `bidirectional` in `kwargs`
            is overridden -- the class is the choice.
        """
        kwargs["bidirectional"] = False
        super().__init__(referent_embedding_size, **kwargs)


class ReceiverTransformerBidirectionalLM(ReceiverTransformerLM):
    def __init__(self, referent_embedding_size, **kwargs):
        """
        The unmasked arm of `ReceiverTransformerLM`: every position reads the
            whole message. Any `bidirectional` in `kwargs` is overridden -- the
            class is the choice.
        """
        kwargs["bidirectional"] = True
        super().__init__(referent_embedding_size, **kwargs)


class ReceiverCrossAttentionLM(nn.Module):
    def __init__(
        self,
        referent_embedding_size,
        **kwargs
    ):
        """
        Read the message with a decoder stack that cross-attends into the
            candidate set, so the meaning it refines is discriminative rather
            than absolute.

        `layers` blocks of self-attention, cross-attention into the candidates,
            and a feedforward. The output is referent-conditioned by
            construction, which is the point of the ordering; see
            docs/architecture.md.

        The candidates arrive already at `d_model` and already normalised.
            This slot used to own the projection and the norm that got them
            there, on the argument that the width a projection targets is a
            property of the consumer and `Receiver` could not know it without
            reaching into slot internals. It does not have to reach: the slot
            *declares* the width, through `referent_input_size`, and `Receiver`
            builds the interface. See `Receiver.__init__` for the contract and
            for the two asymmetries in it.

        Args:
            referent_embedding_size: recorded and not used. This slot declares
                its own `d_model` as the width it wants the candidates at, so
                what `build_models` passes here sizes nothing. Kept on the
                signature because it is the slot contract's shape.
        """
        super().__init__()
        self.referent_embedding_size = referent_embedding_size
        self.token_embedding_size = kwargs["token_embedding_size"]
        self.d_model = kwargs["d_model"]
        self.message_length = kwargs["message_length"]
        self.layers = kwargs["layers"]
        self.heads = kwargs["heads"]
        self.utility_tokens = kwargs["utility_tokens"]
        self.bidirectional = kwargs["bidirectional"]
        self.ff_inner_size = kwargs["ff_inner_size"]
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

        # `decoder=True` counts three sublayers to the block, which is what
        #     these are, so what is passed is the block count and not a multiple
        #     of it. See docs/broccoli.md.
        self.alpha, self.beta = model_util.resolve_residual_scaling(
            kwargs["alpha"], kwargs["beta"], self.layers, decoder=True,
        )

        # Suppressed unless the stack is deep enough for a depth ramp to mean
        #     anything: at one block a linear ramp gives one rate of 0.0.
        self.stochastic_depth = (
            kwargs["stochastic_depth"] if self.layers > 1 else 0.0
        )

        # The message's own adapter, and not an interface in `Receiver`'s
        #     sense: it reads the *token embeddings*, which is this slot's raw
        #     input, rather than an encoded message representation. There is no
        #     message interface upstream of a message encoder, which is what
        #     `message_input_size` returning `None` states.
        self.message_adapter = nn.Linear(
            self.token_embedding_size,
            self.d_model
        )

        # `causal` follows `bidirectional` exactly as the encoder it replaces
        #     did: the message arrives whole, so nothing here masks it left to
        #     right.
        self.message_decoder = transformer_decoder.TransformerDecoder(
            self.message_length,
            # `memory_len` is recorded for the caller and sizes nothing; the
            #     candidate count is a property of the game, not of this module.
            None,
            self.d_model,
            self.layers,
            self.heads,
            # Pinned False, and no longer a config option; every stack here
            #     runs rotary. See docs/broccoli.md.
            absolute_position_embedding=False,
            relative_position_embedding=self.relative_position_embedding,
            # Pinned at 1.0, not configurable -- see `ViT2` for the argument.
            positional_heads=1.0,
            # Derived from the data, not configured separately: this stack
            #     reads the message, so its source is the message length.
            source_size=(self.message_length,),
            # `ff_ratio` None so that `ff_inner_size` is the live knob; note
            #     broccoli's `ViT` resolves the two the other way round.
            ff_ratio=None,
            ff_inner_size=self.ff_inner_size,
            activation=self.activation,
            activation_kwargs=None,
            # Pinned rather than promoted: this argument can never take effect.
            #     Use the inner/outer knobs instead. See docs/broccoli.md.
            ff_dropout=0.0,
            ff_inner_dropout=self.ff_inner_dropout,
            ff_outer_dropout=self.ff_outer_dropout,
            msa_dropout=self.self_attention_dropout,
            cross_attention_dropout=self.cross_attention_dropout,
            stochastic_depth=self.stochastic_depth,
            depthwise_linear_stochastic_depth=self.depthwise_linear_stochastic_depth,
            linear_module=nn.Linear,
            bos_tokens=self.utility_tokens,
            knocking_heads=self.knocking_heads,
            return_bos_tokens=self.return_bos_tokens,
            pre_norm=self.pre_norm,
            post_norm=self.post_norm,
            msa_scaling="d",
            alpha=self.alpha,
            beta=self.beta,
            causal=not self.bidirectional,
            cross_first=False,
        )

    @property
    def referent_input_size(self):
        """
        `d_model`. The candidate set is this stack's cross-attention memory, so
            it has to arrive in the stream's own space.

        The norm `Receiver` puts on top of the interface is not decoration
            here: V is not normed anywhere inside the stack, so without it an
            object could win the attention for being large rather than for
            matching. A post-norm stack normalises its own stream and never its
            memory, and the candidate set is this stack's memory. See
            docs/architecture.md.
        """
        return self.d_model

    @property
    def message_input_size(self):
        """`None`: this slot is the message encoder. See `ReceiverGRULM`."""
        return None

    @property
    def output_size(self):
        return self.d_model

    def forward(
        self,
        messages: torch.Tensor, # (batch, seq_len, token_embedding_size)
        referents: torch.Tensor, # (batch, n_objects, d_model), from `Receiver`
        ) -> torch.Tensor: # -> (batch, message slots, d_model)
        return self.message_decoder(self.message_adapter(messages), referents)

    def reset_parameters(self):
        # Every submodule holding a parameter. The referent projection and its
        #     norm are `Receiver`'s now, and `Receiver.reset_parameters` resets
        #     them; the bug this comment used to mark -- two adapters left out
        #     of a reset and surviving a listener reset -- is the reason that
        #     one is written as a walk over the whole interface container. See
        #     docs/anecdotes.md.
        self.message_adapter.reset_parameters()
        self.message_decoder.reset_parameters()


# --------------------------------------------------------------------------
# Discriminators. Each takes `(referents, message_repr)` and returns
#     `(batch, n_objects)`.
# --------------------------------------------------------------------------

class BilinearDiscriminator(ScoreVolume, nn.Module):
    def __init__(
        self,
        referent_embedding_size,
        message_width,
        **kwargs
    ):
        """
        Score a candidate by a bilinear form: `score = obj_emb.T @ W @ m_emb`,
            read out through `ScoreVolume`.

        The projection has no bias, so the score depends only on the
            relationship between message and object; a bias would add a
            message-independent per-object prior. See docs/architecture.md.

        `bilinear` carries volume as well as direction. Nothing downstream
            divides a rescaling of it back out, so `bilinear_weight_norm` reads
            as volume again -- which is what it meant before `7b10d47`
            standardised the readout, and what jayelm's `CopyListener.compare`
            has always done, though on unnormalised operands.
            `LegacyBilinearGRUComparer`, in tests/test_receiver_slots.py, still
            records that arrangement. The matrix sharing the volume with
            `log_score_scale` is not a problem the way two scalars would be: a
            320x320 matrix under Adam spends its step turning and only a small
            fraction of it radially, which is why the scalar exists (it moved
            59% where the matrix managed 1.3%), and the scalar is the fast
            path rather than a competing one.

        Args:
            referent_embedding_size: the width the comparison runs at, and the
                one the `1/sqrt(d)` calibration is taken over. Declared upstream
                as `referent_input_size`, so `Receiver` projects the backbone's
                output to it. `build_models` passes
                `[receiver_language_model] d_model`.
            message_width: the width the message is read at, declared upstream
                as `message_input_size`. `build_models` passes the language
                model's `output_size`, which makes that interface square.

        **This module owns no norms.** Both operands arrive normalised, because
            `Receiver` delivers every input to a slot in a stated distribution
            -- `dropout(norm(adapter(referents)))` and `norm(adapter(message))`
            -- and this module declares the two widths it wants them at. The
            two `nn.LayerNorm`s that used to live here did the same job one
            stage later.

        `scale_score` and `bias_score` arrive in `kwargs` from
            `[receiver_discriminator]` and both default true, which is today's
            arithmetic exactly. Each builds one scalar of the `ScoreVolume`
            readout and nothing else. **Neither reaches the `1/sqrt(d)`
            calibration**, which is unconditional -- see `forward`.

            They replace a single `normalise_score`, which gated the calibration
            and both scalars together and, before the interface hoist, this
            module's two operand norms as well. That third job is what made the
            flag's `false` an exact revert to jayelm's unnormalised
            `CopyListener.compare`; the norms are `Receiver`'s interfaces now,
            unconditional and part of delivering an input at a declared width,
            so `false` had stopped being a revert and had started being an
            arrangement -- normalised operands, uncalibrated score -- that
            nobody designed and no run has used. Rather than reach back into
            `Receiver` from a discriminator's config key, which would
            reintroduce exactly the coupling the hoist removed, the calibration
            is design and the two scalars are the configuration. See
            docs/channel.md.
        """
        super().__init__()
        self.referent_embedding_size = referent_embedding_size
        self.message_width = message_width

        self._init_score_volume(
            kwargs.get("scale_score", True),
            kwargs.get("bias_score", True),
        )

        self.bilinear = nn.Linear(
            self.message_width,
            self.referent_embedding_size,
            bias=False
        )

    @property
    def referent_input_size(self):
        """
        The width the candidates are compared at, and the one the `1/sqrt(d)`
            calibration is taken over. Set from the constructor argument, which
            `build_models` takes from `[receiver_language_model] d_model`.
        """
        return self.referent_embedding_size

    @property
    def message_input_size(self):
        """
        The width `bilinear` reads the message at -- the language model's
            `output_size`, as `build_models` passes it. Declared rather than
            assumed, so the interface upstream is sized from this module's
            statement of what it wants and not from a guess about the encoder.
        """
        return self.message_width

    def forward(
        self,
        referents: torch.Tensor, # (batch, n_objects, d_embedding)
        message_repr: torch.Tensor # (batch, slots, message_width)
        ) -> torch.Tensor: # -> (batch, n_objects)
        """
        The last slot is the identity for `ReceiverGRULM`, which returns one and
            has already taken its final state, and for `ReceiverTransformerLM`,
            which returns one because it has already pooled. For a stack that returns one
            position per symbol it is the reserved EOS position: the speaker's
            messages are fixed length, so EOS is positionally determined and
            carries no information of its own, and its input embedding is
            therefore a constant learned vector -- a CLS query in all but name.
            A causal stack reaches it having read the whole message and a
            bidirectional one does too, which a mean over slots does not
            improve on: it dilutes the readout across every symbol.

        Robust to prepended utility tokens, which do not move the last position.
        """
        message_embeddings = message_repr[:, -1, :]

        projected = self.bilinear(message_embeddings)
        scores = torch.einsum("ijh,ih->ij", (referents, projected)) # (batch, n_objects)

        # The calibration, and it is **unconditional**: it is exact rather than
        #     approximate because both operands arrive normalised -- which they
        #     do by the interface contract rather than by a norm this module
        #     owns, and so under every configuration. Each is at unit variance,
        #     so `|r| = |m| = sqrt(d)`, and with `nn.Linear`'s default init --
        #     uniform on `+/- 1/sqrt(d)`, standard deviation `1/sqrt(3d)` -- the
        #     score's standard deviation at init is `sigma_w * d = sqrt(d / 3)`.
        #     Dividing by `sqrt(d)` leaves `1/sqrt(3)` = 0.577 at every width
        #     and under every backbone, which is what makes the opening a number
        #     this repo can state rather than measure per rung. Removing it is
        #     not a rung and never was a hypothesis: measured on this module,
        #     the uncalibrated score opens at sd 9.6 and BCE 3.81 at d = 256,
        #     and sd 18.4 and BCE 6.87 at d = 1024, against `ln 2` = 0.693.
        #     There is no configuration that turns it off.
        #
        # `log_score_scale` opens at 0, so the readout opens there too. BCE on
        #     a random map at that spread is 0.725 against `ln 2` = 0.693; at
        #     unit spread it would be 0.804. The gentler opening is the point of
        #     leaving the scalar at 1.0 rather than calibrating it to `sqrt(3)`.
        return self.readout(scores / math.sqrt(self.referent_embedding_size))

    def reset_parameters(self):
        # Two things, and there is nothing else here to miss: the operand norms
        #     moved to `Receiver`, which resets its own interfaces.
        self.bilinear.reset_parameters()
        self.reset_score_volume()


# --------------------------------------------------------------------------
# The interfaces. One per (slot, input) pair that a slot declares a width for.
#
# `model_util.LinearInterface` is the whole of what one is -- a linear map to
#     the declared width, an affine-free `LayerNorm`, and a mask on the referent
#     side. It is the speaker's `adapter` too; see that class for the rule and
#     for the two asymmetries in it.
# --------------------------------------------------------------------------

# The keys of `Receiver.interfaces`, in build order -- which is also RNG order,
#     so moving one moves every parameter drawn after it.
LANGUAGE_MODEL_REFERENTS = "language_model_referents"
DISCRIMINATOR_REFERENTS = "discriminator_referents"
DISCRIMINATOR_MESSAGE = "discriminator_message"


def build_interfaces(
    feature_size, message_width, language_model, discriminator, dropout
):
    """
    One `model_util.LinearInterface` per input a slot declares a width for,
        keyed by
        `(slot, input)`.

    A function rather than three lines in `Receiver.__init__` so that the test
        shim in `tests/_bootstrap.py`, which stands in for `Receiver` from the
        backbone features inwards, mirrors this arrangement by calling it
        rather than by reproducing it. A shim that reproduces the thing it
        stands in for is a shim that will one day disagree with it.

    Args:
        feature_size: the backbone's `final_feat_dim`, the input width of every
            referent interface
        message_width: the language model's `output_size`, the input width of
            every message interface
        language_model: the message-encoding slot
        discriminator: the scoring slot
        dropout: the mask rate on referent interfaces; message interfaces are
            never masked

    Returns:
        An `nn.ModuleDict`. A slot that declares `None` for an input, or that
            declares nothing at all, gets no entry -- see `Receiver.__init__`
            for what that means at the call site.
    """
    interfaces = nn.ModuleDict()

    for key, module, input_size, attribute in (
        (
            LANGUAGE_MODEL_REFERENTS,
            language_model,
            feature_size,
            "referent_input_size",
        ),
        (
            DISCRIMINATOR_REFERENTS,
            discriminator,
            feature_size,
            "referent_input_size",
        ),
        (
            DISCRIMINATOR_MESSAGE,
            discriminator,
            message_width,
            "message_input_size",
        ),
    ):
        # `getattr` with a default rather than a bare attribute read: the
        #     contract is "absent or None means this module does not take that
        #     input", so a slot written before this existed declares nothing and
        #     gets nothing, rather than raising at build time inside a module it
        #     has no business knowing about.
        declared = getattr(module, attribute, None)

        if declared is None:
            continue

        referents = attribute == "referent_input_size"

        interfaces[key] = model_util.LinearInterface(
            input_size,
            declared,
            bias=not referents,
            dropout=dropout if referents else None,
        )

    return interfaces


def through(interfaces, key, x):
    """
    Deliver `x` through the named interface, or hand back `None` where the slot
        declared it takes no such input.

    Shared with the test shim for the same reason `build_interfaces` is, and
        read at every call site so that "no interface" and "no tensor" cannot
        come apart.
    """
    # `in` rather than a `.get`: `nn.ModuleDict` is not a `dict` and has no
    #     `get`.
    return interfaces[key](x) if key in interfaces else None


class Receiver(nn.Module):
    def __init__(
        self,
        feature_model,
        feature_size,
        token_embedding_module,
        language_model,
        discriminator,
        dropout=0.1,
    ):
        """
        An agent that scores a set of candidate referents against a message.

        Args:
            feature_model: produces embeddings from referents
            feature_size: the backbone's `final_feat_dim`. Passed rather than
                read off `feature_model` so that a test can compose the
                listener's plumbing around something that is not a backbone.
            token_embedding_module: the message's embedding table
            language_model: encodes the message, `(batch, slots, width)`
            discriminator: scores the candidates from that encoding
            dropout: the rate on every referent interface. Counterpart to the
                speaker's `prototype_dropout`. Note there is no separate
                `vision_dropout` here, unlike `Sender`: it would mask the same
                tensor with nothing but a reshape between the two. See
                docs/architecture.md.

        **The slots declare, `Receiver` delivers.** Each slot exposes
            `referent_input_size` and `message_input_size`; `None` means the
            slot does not take that input at all, and `Receiver` then builds
            nothing and passes `None` in its place. For every declared width it
            builds a `model_util.LinearInterface` -- see that class for the
            rule, for the two asymmetries in it, and for why the norms are
            unconditional.

            This inverts the arrangement it replaces, where each consumer owned
            its own projection and the widths were decided four different ways:
            `Receiver` held one adapter of its own upstream of the lot, each
            slot held its own projection and norm, `BilinearDiscriminator` held
            no projection at all and took the referent width as its own output
            width, and the since-removed attention discriminator consumed the
            referents *twice at two widths* -- a second consumer that `Receiver`
            could not see, which is what made the arrangement hard to change. The objection to sharing
            was that `Receiver` would have to work out which slots wanted which
            width, which is reaching into slot internals. Under a declaration it
            works nothing out: it reads what the slot states.

        **One mask per referent interface, drawn independently.** The masks used
            to be one mask, applied once here and handed to both slots, on the
            argument that a per-slot mask would regularise the listener at a
            rate no config key names. That argument was about masking *one*
            tensor twice. These are two tensors: each slot's own projected copy
            of the referents, each masked once at the rate `[receiver] dropout`
            names. Independence is what makes them the two masks the rate
            describes rather than a correlated pair.

            The mask is element-wise over `(batch, n_objects, features)`, so it
            removes features within each candidate rather than removing whole
            candidates -- which would leak the label ordering.

        **The message operand is never masked.** See `model_util.LinearInterface`.
        """
        super().__init__()
        self.feature_model = feature_model
        self.feature_size = feature_size
        self.token_embedding = token_embedding_module
        self.language_model = language_model
        self.discriminator = discriminator
        self.dropout = dropout

        # One container rather than three attributes, so `MODULE_GROUPS` can
        #     select the lot with one entry: `receiver_adapter` keeps its name,
        #     its `[optimiser.module_lr]` key and its `clip_*` column, and
        #     `GROUP_NAMES` does not change shape. The alternative -- three
        #     group names -- cascades into `train.py`'s header,
        #     `validate_config`'s key check and `DEFAULT.toml`, for no gain.
        #
        # An `nn.ModuleDict` and not a list: which interface an entry is has to
        #     survive a slot that declares no width, and a positional container
        #     would renumber itself when one goes missing.
        self.interfaces = build_interfaces(
            feature_size,
            language_model.output_size,
            language_model,
            discriminator,
            dropout,
        )

    def forward(self, referents, messages):
        batch_size = referents.shape[0]
        n_obj = referents.shape[1]
        rest = referents.shape[2:]

        # Embed the referents, once. Everything below is a view of these
        #     features through one interface or another.
        referents_flat = referents.view(batch_size * n_obj, *rest)
        features = self.feature_model(referents_flat).view(batch_size, n_obj, -1)

        # Embed the messages
        messages = messages @ self.token_embedding.weight

        message_repr = self.language_model(
            messages,
            through(self.interfaces, LANGUAGE_MODEL_REFERENTS, features),
        )

        return self.discriminator(
            # A mask of its own, drawn here rather than shared with the
            #     language model's. See `__init__`.
            through(self.interfaces, DISCRIMINATOR_REFERENTS, features),
            through(self.interfaces, DISCRIMINATOR_MESSAGE, message_repr),
        )

    def reset_parameters(self):
        self.feature_model.reset_parameters()
        self.token_embedding.reset_parameters()
        self.language_model.reset_parameters()
        self.discriminator.reset_parameters()

        # A walk over the container rather than a list of attribute names, so
        #     an interface cannot be added and left out of the reset. That is
        #     not hypothetical: two of the adapters this replaces were once
        #     missing from `ReceiverCrossAttentionLM.reset_parameters` and
        #     survived a `receiver_reset_interval` reset. See docs/anecdotes.md.
        for interface in self.interfaces.values():
            interface.reset_parameters()
