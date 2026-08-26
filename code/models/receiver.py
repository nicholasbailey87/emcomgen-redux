"""
Listener models

The listener is two swappable slots, mirroring the speaker's
`prototyper` / `language_model` split:

    language_model  encodes the message           -> (batch, slots, width)
    discriminator   scores the candidates from it -> (batch, n_objects)

`Receiver` owns the composition, the token embedding, and one dropout. Both
slots are named in `[receiver]` and configured from `[receiver_language_model]`
and `[receiver_discriminator]`, which makes four legal combinations:

                             BilinearDiscriminator  AttentionDiscriminator
    ReceiverGRULM            the historical baseline        new
    ReceiverCrossAttentionLM           new             the attention rung

The two new cells are the point. Before this split, one `comparer` key chose
both halves at once, so a rung that swapped the GRU comparer for the
cross-attention one changed the message encoder *and* the comparison, and
"attention helps" could not be attributed to either. See docs/architecture.md.

**Exactly one message encoder, always.** `AttentionDiscriminator` carries an
internal bilinear path, and that is a second *comparison*, not a second
encoder: it reads whatever the language model produced. No configuration builds
two encoders, and if one ever looks necessary the slot contract is wrong.
"""

import torch
import torch.nn as nn

import broccoli

from . import model_util
from . import transformer_decoder

# Mirrors `sender.LAYER_NORM_EPS`, and load-bearing for the same reason: below
#     the 1e-5 default the normaliser quietly stops normalising and the score's
#     magnitude goes back to the backbone. See docs/channel.md.
LAYER_NORM_EPS = 1e-12

# Every broccoli module below is constructed with its full argument list, even
#     where an argument is inert under the current settings, because broccoli's
#     defaults are not a stable interface. See docs/broccoli.md.


def standardise(scores):
    """
    Per game: remove the mean over candidates and scale to unit spread.

    Every discriminator's readout runs through this. It is the shape half of the
        shape/volume split both agents now have -- `logit_spread` against
        `logit_scale` on the speaker, this against `score_scale` here -- so a
        score's magnitude is a learned scalar and nothing else, at any width and
        under any backbone.

    Note what it does *not* pin. The location and the scale of a game's scores
        go; the shape stays free. `train.py` decides on `scores > 0`, so the
        threshold is the game's own mean, which is where a listener facing
        balanced candidates would put it anyway -- but a listener wanting one
        target out of twenty scores it at `sqrt(19) = 4.359` and the other
        nineteen at `-1/sqrt(19) = -0.229`, which is mean-zero and unit-spread
        and puts exactly one candidate above the threshold. The centring is an
        opening, not a constraint. It does mean `train_acc` is not comparable
        across the commit that introduced it.

    Also used by `AttentionDiscriminator`'s telemetry, on each branch
        separately, where it is what makes `path_agreement` a Pearson r and lets
        `mix_share` compare two unstandardised branches like with like.

    `unbiased=False` because this is a population statistic over the candidate
        set, not an estimate from a sample of it, and the clamp keeps a game
        whose candidates happen to score identically from dividing by zero. A
        clamp is safe here in a way it is not on a parameter: nothing learns
        through this bound.
    """
    centred = scores - scores.mean(1, keepdim=True)
    spread = centred.std(dim=1, keepdim=True, unbiased=False)
    return centred / spread.clamp(min=1e-6)


class ScoreVolume:
    """
    The listener's one degree of freedom over how loudly it states a conclusion,
        and the counterpart of the speaker's `GumbelChannel.logit_scale`. See
        docs/architecture.md.

    A mixin rather than a submodule, for the same reason `GumbelChannel` is one:
        `log_score_scale` stays registered on the discriminator itself, so the
        `state_dict` key is the one `split_out_parameter` matches by suffix and
        the one earlier checkpoints were written against.

    The readout is `score_scale * standardise(scores)`, applied through
        `model_util.scale_without_attenuating` so the scale is absent from the
        backward pass into everything upstream. That is the whole point of this
        module and the reasoning is in the helper's docstring: BCE at `p = 0.5`
        is right to turn a listener down on a message carrying nothing, and a
        scalar at the front of the score turns the *speaker* down at the same
        time, which is what keeps the message carrying nothing.

    With the readout normalising, the volume cannot live anywhere else. A
        `BilinearDiscriminator` standing alone is exactly scale-invariant in
        `bilinear.weight`, so that matrix now learns direction only -- read
        `bilinear_weight_norm` as drift rather than as volume, and note that
        with `weight_decay = 0.0` a scale-invariant weight's norm can only grow
        while its effective learning rate, `~ lr * sqrt(d) / ||W||` under Adam,
        decays with it. Inside `AttentionDiscriminator` the branch magnitudes
        still set the mix, so there the norms stay load-bearing.
    """

    def _init_score_volume(self, learns_score_scale=True):
        """
        Call from `__init__` where the parameter should be created: creation
            order fixes which RNG draw every later parameter gets.

        `learns_score_scale=False` is passed only from inside
            `AttentionDiscriminator`, to the bilinear path it composes, because
            that path's output is standardised by the readout downstream of it
            and `standardise(s * u) == standardise(u)` exactly for positive `s`.
            A scale there would take identically zero gradient, report a
            constant 1.0, and sit in an elevated learning-rate group doing
            nothing. Absent rather than frozen, so `split_out_parameter`'s
            suffix match sees the truth.
        """
        self.learns_score_scale = learns_score_scale

        if self.learns_score_scale:
            # Stored as its log so `exp` keeps it strictly positive: gradient
            #     descent cannot walk a volume through zero and out the far
            #     side. Opens at 1.0, and after `standardise` that is a genuine
            #     unit-spread opening rather than a number whose meaning depends
            #     on the backbone.
            self.log_score_scale = nn.Parameter(torch.zeros(()))

    @property
    def score_scale(self):
        """
        The multiplier applied to the standardised scores, always positive. Read
            here rather than exponentiating at the use site so `forward` and the
            metrics column cannot drift apart.
        """
        return self.log_score_scale.exp()

    def readout(self, scores):
        """
        Standardise per game, then apply the volume without attenuating the
            gradient reaching the message.

        A discriminator built with `learns_score_scale=False` returns the
            standardised scores unscaled, which is what its caller wants: the
            caller's own readout is downstream and would have divided any scale
            back out.
        """
        standardised = standardise(scores)

        if not self.learns_score_scale:
            return standardised

        return model_util.scale_without_attenuating(
            standardised, self.score_scale
        )

    def reset_score_scale(self):
        """
        Put the volume back to its 1.0 opening, so a reset does not leave a
            trained confidence behind a fresh listener.
        """
        if self.learns_score_scale:
            with torch.no_grad():
                self.log_score_scale.zero_()


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

        Returns a length-1 sequence rather than a bare vector so either
            discriminator can consume either language model.
            `BilinearDiscriminator` means over that axis, which is the identity
            here; `AttentionDiscriminator` takes it as cross-attention memory,
            where a length-1 memory is perfectly legal.
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
            #     `[receiver] dropout`, which masks the referents once in
            #     `Receiver` -- see docs/architecture.md.
            dropout=0.0,
            bidirectional=self.bidirectional
        )

    @property
    def output_size(self):
        """
        The width the discriminator must adapt from. Read here rather than
            recomputed at the call site so the two cannot drift apart.
        """
        return self.d_model * 2 if self.bidirectional else self.d_model

    def forward(
        self,
        messages: torch.Tensor, # (batch, seq_len, token_embedding_size)
        referents: torch.Tensor, # (batch, n_objects, d_embedding), ignored
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

        This slot owns its own referent projection and norm rather than taking
            them from `Receiver`. The width a projection targets is a property
            of the consumer, and pairing this with `AttentionDiscriminator`
            gives two consumers at possibly different widths -- so a shared
            adapter would have `Receiver` reaching into slot internals to work
            out which of them needs one, which is the coupling this split
            exists to remove. The duplication costs one `d_model x feat` matrix
            in the one combination where both slots want it.
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

        # `bias=False`, and load-bearing: `referent_layer_norm` can only divide
        #     the backbone's scale out exactly if what reaches it is homogeneous
        #     in the input. See docs/architecture.md.
        self.referent_adapter = nn.Linear(
            self.referent_embedding_size,
            self.d_model,
            bias=False
        )

        # Parameter-free, and what it fixes is *per-object* magnitude: V is not
        #     normed anywhere, so without this an object can win the attention
        #     for being large rather than for matching. A post-norm stack
        #     normalises its own stream and never its memory, and the candidate
        #     set is this stack's memory. See docs/architecture.md.
        self.referent_layer_norm = nn.LayerNorm(
            self.d_model, elementwise_affine=False, eps=LAYER_NORM_EPS
        )

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
    def output_size(self):
        return self.d_model

    def forward(
        self,
        messages: torch.Tensor, # (batch, seq_len, token_embedding_size)
        referents: torch.Tensor, # (batch, n_objects, d_embedding)
        ) -> torch.Tensor: # -> (batch, message slots, d_model)
        adapted = self.referent_layer_norm(self.referent_adapter(referents))
        return self.message_decoder(self.message_adapter(messages), adapted)

    def reset_parameters(self):
        # Every submodule holding a parameter, including the two adapters (which
        #     were missing here once) and the parameter-free norm. See
        #     docs/anecdotes.md.
        self.referent_adapter.reset_parameters()
        self.referent_layer_norm.reset_parameters()
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
        score_scale=True,
        **kwargs
    ):
        """
        Score a candidate by a bilinear form: `score = obj_emb.T @ W @ m_emb`,
            read out through `ScoreVolume`.

        The projection has no bias, so the score depends only on the
            relationship between message and object; a bias would add a
            message-independent per-object prior. See docs/architecture.md.

        `bilinear` learns a *direction* and nothing else. The readout
            standardises per game, so the module is exactly scale-invariant in
            this weight: scaling it cannot change the score, the decision or the
            loss. That is a reversal of the arrangement immediately before,
            where the volume lived in this matrix on the argument that a matrix
            has no cheap move downwards -- true, and the problem: it had no
            usable move at all, 1.3% of its norm over thirty epochs against the
            scalar it replaced travelling 59%. jayelm's `CopyListener.compare`
            does hold its volume this way, on unnormalised operands, and
            `LegacyBilinearGRUComparer`, in tests/test_receiver_slots.py,
            still records what that was.

        Args:
            referent_embedding_size: width of the backbone's output
            message_width: the language model's `output_size`
            score_scale: build the learnable volume. False only from inside
                `AttentionDiscriminator`; see `ScoreVolume._init_score_volume`.
        """
        super().__init__()
        self.referent_embedding_size = referent_embedding_size
        self.message_width = message_width

        self._init_score_volume(score_scale)

        self.bilinear = nn.Linear(
            self.message_width,
            self.referent_embedding_size,
            bias=False
        )

        # The two operands of the dot product, normalised per example over the
        #     feature axis so the score's magnitude is not inherited from the
        #     vision model. Neither norm is affine. Note they are in *different*
        #     spaces: the message is normalised before `bilinear` reads it, so
        #     that norm is `message_width` wide, and the referent one is in
        #     referent space. See docs/architecture.md.
        self.referent_layer_norm = nn.LayerNorm(
            self.referent_embedding_size,
            elementwise_affine=False,
            eps=LAYER_NORM_EPS,
        )
        self.message_layer_norm = nn.LayerNorm(
            self.message_width,
            elementwise_affine=False,
            eps=LAYER_NORM_EPS,
        )

    def forward(
        self,
        referents: torch.Tensor, # (batch, n_objects, d_embedding)
        message_repr: torch.Tensor # (batch, slots, message_width)
        ) -> torch.Tensor: # -> (batch, n_objects)
        """
        The last slot is the identity for `ReceiverGRULM`, which returns one and
            has already taken its final state. For a stack that returns one
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

        # Normalised *before* the projection. The scaling half of this norm is
        #     now redundant -- the message is one vector per game, so any factor
        #     on it multiplies all the candidates' scores equally and the
        #     readout's `standardise` divides it straight back out. The centring
        #     is not redundant, and neither norm is free: both set where
        #     `bilinear` starts, and it is only the direction that has to be
        #     learned from there.
        projected = self.bilinear(self.message_layer_norm(message_embeddings))

        # This one *is* load-bearing after the readout. It normalises each
        #     candidate independently, so it changes the scores' relative order
        #     and not just their common scale -- without it a large referent is
        #     read loudly for being large, which `standardise` could not undo.
        referents = self.referent_layer_norm(referents)

        scores = torch.einsum("ijh,ih->ij", (referents, projected)) # (batch, n_objects)

        # There was a `1/sqrt(referent_embedding_size)` here, to make the score
        #     open at a width-independent magnitude with `bilinear` at the
        #     default init. `standardise` divides out any constant factor
        #     exactly, so it now opens at unit spread at every width by
        #     construction and the calibration has nothing left to do.
        return self.readout(scores)

    def reset_parameters(self):
        self.bilinear.reset_parameters()
        self.reset_score_scale()
        # No-ops while the two norms are parameter-free, and listed anyway so
        #     that turning `elementwise_affine` back on cannot leave a reset
        #     listener holding trained gains.
        self.referent_layer_norm.reset_parameters()
        self.message_layer_norm.reset_parameters()


class AttentionDiscriminator(ScoreVolume, nn.Module):
    def __init__(
        self,
        referent_embedding_size,
        message_width,
        **kwargs
    ):
        """
        Score the candidates with a decoder stack that reads the encoded
            message as memory, mixed with a bilinear score over the same
            message:

            score = readout((1 - a) * bilinear + a * attention) + bias
            a     = mix_floor + (1 - mix_floor) * sigmoid(mix_logit)

        `referent_decoder` runs `layers` blocks: cross-attention into the
            message *first*, then the candidates read each other, then a
            feedforward -- so the message crosses into the scored stream once
            per block rather than once in total, and the candidates compare
            message-informed representations. `decision` is a plain
            `nn.Linear(d_model, 1)`.

        **Why the mix.** The attention path alone does not bootstrap. Under a
            nuisance level where the bilinear comparison reaches 0.938 it sits
            at 0.469 with its polarity tag barely moving, because at
            initialisation nothing in the pair yet looks like a pattern and
            uniformity is the cheapest thing a listener can do. This gives the
            speaker a gradient worth having from step zero and lets attention
            take over if it earns it -- the recipe `AttentionPrototyper`
            already follows: open at the simple behaviour and depart only if it
            pays. See docs/architecture.md.

        **Where the volume lives.** One `log_score_scale`, from `ScoreVolume`,
            downstream of the mix -- the same readout `BilinearDiscriminator`
            uses, so there is one volume mechanism on the listener rather than
            two. The composed bilinear path is built with `score_scale=False`
            because it now feeds this readout instead of being one.

        `standardise` runs on the *mixed* score and not on each branch. That is
            the difference from the arrangement this replaces, where
            standardising per branch made `a` mean composition exactly and left
            `log_mix_scale` as the only magnitude. Standardising after the mix
            gets the single volume knob without pinning the branches to equal
            spread, so a branch can still be loud or quiet on its own and
            `mix_share` against `mix_alpha` still measures it -- the first is
            the share the score is actually made of, the second the share
            `mix_logit` asked for.

        Neither is this the change that once stopped four rungs learning. That
            was the attention readout pinned to a *fixed gain*, which cannot go
            quiet while the message is still noise and so forces a listener to
            commit before there is anything to commit to (see
            docs/anecdotes.md). This listener can go as quiet as BCE asks it to,
            and going quiet no longer costs the speaker anything, which is what
            `model_util.scale_without_attenuating` is for. The bilinear path
            still carries the decision through the opening, so nothing has to be
            confident early.

        The floor is in the parameterisation and **never a `clamp`**:
            `clamp`'s gradient is zero below the bound, so a mix that drifted
            under the floor would weld there permanently. With the floor in
            place the attention path always contributes and so always receives
            gradient.

        The bilinear path is a second *comparison*, not a second encoder: it
            reads whatever `message_repr` the language model handed over. See
            this module's docstring for why that invariant matters.

        Args:
            referent_embedding_size: width of the backbone's output
            message_width: the language model's `output_size`
        """
        super().__init__()
        self.referent_embedding_size = referent_embedding_size
        self.message_width = message_width

        self._init_score_volume()

        self.d_model = kwargs["d_model"]
        self.layers = kwargs["layers"]
        self.heads = kwargs["heads"]
        self.ff_inner_size = kwargs["ff_inner_size"]
        self.activation = model_util.get_activation(kwargs["activation"])
        self.pre_norm = kwargs["pre_norm"]
        self.post_norm = kwargs["post_norm"]
        self.knocking_heads = kwargs["knocking_heads"]
        self.depthwise_linear_stochastic_depth = kwargs[
            "depthwise_linear_stochastic_depth"
        ]
        self.ff_inner_dropout = kwargs["ff_inner_dropout"]
        self.ff_outer_dropout = kwargs["ff_outer_dropout"]
        self.self_attention_dropout = kwargs["self_attention_dropout"]
        self.cross_attention_dropout = kwargs["cross_attention_dropout"]
        self.mix_floor = kwargs["mix_floor"]
        self.mix_logit_init = kwargs["mix_logit_init"]

        if not 0.0 <= self.mix_floor < 1.0:
            raise ValueError(
                "`mix_floor` is the attention path's minimum share of the "
                f"score and must be in [0, 1), got {self.mix_floor}."
            )

        self.alpha, self.beta = model_util.resolve_residual_scaling(
            kwargs["alpha"], kwargs["beta"], self.layers, decoder=True,
        )

        self.stochastic_depth = (
            kwargs["stochastic_depth"] if self.layers > 1 else 0.0
        )

        # This slot's own projection and norm; see `ReceiverCrossAttentionLM`
        #     for why they are not shared with the language model's.
        self.referent_adapter = nn.Linear(
            self.referent_embedding_size,
            self.d_model,
            bias=False
        )
        self.referent_layer_norm = nn.LayerNorm(
            self.d_model, elementwise_affine=False, eps=LAYER_NORM_EPS
        )

        # The memory this stack reads, brought to `d_model` from whatever width
        #     the language model works at -- 2 * d_model for a bidirectional
        #     GRU, its own d_model for the cross-attention encoder, and there is
        #     no arithmetic that makes those agree. Normed for the same reason
        #     the referents are: a post-norm stack normalises its own stream and
        #     never its memory, so nothing else would divide out a GRU state's
        #     magnitude.
        self.memory_adapter = nn.Linear(self.message_width, self.d_model)
        self.memory_layer_norm = nn.LayerNorm(
            self.d_model, elementwise_affine=False, eps=LAYER_NORM_EPS
        )

        # `causal=False` is not negotiable, and `relative_position_embedding`
        #     is False for the same reason: referent order is the label vector,
        #     so anything able to index its own sequence axis could ignore the
        #     message. `test_no_stage_can_read_the_referent_ordering` pins it.
        #
        # `cross_first`, so what the self-attention compares has already been
        #     informed by the message. That self-attention is the only stage at
        #     which a score can depend on the rest of the set, and the only
        #     route to the concept game's clustering shortcut.
        self.referent_decoder = transformer_decoder.TransformerDecoder(
            # `seq_len` sizes the causal mask, which is off here, and the
            #     absolute position embedding, which is not built.
            None,
            # `memory_len` sizes nothing, and could not be stated here in any
            #     case: the message representation is one position long from a
            #     GRU and `message_length` long from the decoder stack.
            None,
            self.d_model,
            self.layers,
            self.heads,
            absolute_position_embedding=False,
            relative_position_embedding=False,
            positional_heads=1.0,
            source_size=None,
            ff_ratio=None,
            ff_inner_size=self.ff_inner_size,
            activation=self.activation,
            activation_kwargs=None,
            ff_dropout=0.0,
            ff_inner_dropout=self.ff_inner_dropout,
            ff_outer_dropout=self.ff_outer_dropout,
            msa_dropout=self.self_attention_dropout,
            cross_attention_dropout=self.cross_attention_dropout,
            stochastic_depth=self.stochastic_depth,
            depthwise_linear_stochastic_depth=self.depthwise_linear_stochastic_depth,
            linear_module=nn.Linear,
            bos_tokens=0,
            knocking_heads=False,
            return_bos_tokens=False,
            pre_norm=self.pre_norm,
            post_norm=self.post_norm,
            msa_scaling="d",
            alpha=self.alpha,
            beta=self.beta,
            causal=False,
            cross_first=True,
        )

        # A weight whose magnitude is free, and no bias. The bias was removed
        #     once and restored, and is now removed again for a different and
        #     narrower reason: it adds the same constant to every candidate, and
        #     the readout's centring subtracts the mean over candidates, so it
        #     would take identically zero gradient. The score's one offset is
        #     `mix_bias` below, downstream of the readout where it survives. See
        #     docs/anecdotes.md.
        #
        # It reads straight off the referent stack's last post-norm, which is an
        #     `RMSNorm` and so already equalises the candidates' lengths -- no
        #     object can be read loudly for being large.
        self.decision = nn.Linear(self.d_model, 1, bias=False)

        # The other path. A whole `BilinearDiscriminator`, composed rather than
        #     reimplemented, so the `a -> mix_floor` limit of this module is
        #     the module that was measured bootstrapping and not a lookalike.
        #     It reads `message_repr`; it owns no encoder.
        #
        # Built without a volume: this module's readout standardises downstream
        #     of it, and a scale upstream of a standardise is annihilated
        #     exactly. See `ScoreVolume._init_score_volume`.
        self.bilinear = BilinearDiscriminator(
            self.referent_embedding_size,
            self.message_width,
            score_scale=False,
        )

        # `mix_logit_init` -4.0 puts `a` at 0.116 for the default floor of
        #     0.1 -- open essentially at the bilinear comparison, with the
        #     attention path present enough to be learning.
        self.mix_logit = nn.Parameter(
            torch.tensor(float(self.mix_logit_init))
        )

        # The score's offset, and the only thing downstream of the readout.
        #     It has to be downstream: an offset applied before `standardise`
        #     is a constant across candidates and the centring removes it. That
        #     is also why `decision` above lost its bias.
        #
        # This is where a `log_mix_scale` used to sit, carrying the volume for
        #     the whole module. There is again exactly one volume scalar here,
        #     but it is `ScoreVolume.log_score_scale` -- the same one the
        #     bilinear arm has, under the same config key, applied the same way.
        self.mix_bias = nn.Parameter(torch.zeros(()))

        # Metrics only: set on every `forward`, read by `train.py`. See
        #     docs/measurement.md.
        #
        # The raw mixing weight. Bounded below by `mix_floor` and above by 1.
        #     Note this is the *weight*, not the share: with the branches
        #     unstandardised a loud branch can dominate a heavily-weighted quiet
        #     one, so "was attention used" is `mix_share` below and this is the
        #     parameter that would like it to be.
        self.mix_alpha = float("nan")

        # The realised share of the attention path, measured from the branches
        #     standardised per game -- which is what `mix_alpha` alone would
        #     mean if `forward` standardised them separately rather than
        #     standardising the mix.
        self.mix_share = float("nan")

        # `corr(attention_hat, bilinear_hat)` within a game. Necessary because
        #     an attention path that is never used and one that has learned to
        #     imitate the bilinear path look identical from accuracy and from
        #     `mix_alpha` alone, and they are different findings.
        self.path_agreement = float("nan")

        # The standard deviation of the scores. Now `score_scale` by
        #     construction -- the readout standardises and then multiplies, so
        #     this measures the same thing the `score_scale` column does, up to
        #     `mix_bias` and the pooling over the batch. Kept because it is
        #     defined on any discriminator with a `forward`, where the scale is
        #     a parameter this one happens to have. Read `score_scale`.
        self.decision_spread = float("nan")

        # Excess kurtosis of the scores. Negative means bimodal, which is what
        #     discriminating looks like; sustained positive alongside chance
        #     accuracy is a listener with nothing to say.
        #
        # The column carrying information the others do not, now that the
        #     readout fixes the spread: kurtosis is invariant to the scale, so
        #     it reads the *shape* `standardise` leaves free.
        self.decision_kurtosis = float("nan")

    @property
    def mix_weight(self):
        """
        The weight on the attention path, in `[mix_floor, 1)`. Read here rather
            than recomputed at the use site so `forward` and the metrics column
            cannot drift apart.

        Not the same thing as the attention path's *share* of the score, which
            it was while `forward` standardised both branches. `mix_share`
            measures that.
        """
        return (
            self.mix_floor
            + (1.0 - self.mix_floor) * torch.sigmoid(self.mix_logit)
        )

    def forward(
        self,
        referents: torch.Tensor, # (batch, n_objects, d_embedding)
        message_repr: torch.Tensor # (batch, slots, message_width)
        ) -> torch.Tensor: # -> (batch, n_objects)
        """
        The two paths are mixed at their own magnitudes, and the *mix* is
            standardised by the readout. `standardise` used to run on each of
            them separately here, which made `mix_logit` mean composition
            exactly; running it after the mix keeps one volume scalar without
            pinning the branches to equal spread, so a branch can still escape
            being learned by turning itself down -- watch `mix_share` against
            `mix_alpha`. It still runs per branch in the telemetry block below,
            where that is what `path_agreement` and `mix_share` need.
        """
        adapted = self.referent_layer_norm(self.referent_adapter(referents))
        memory = self.memory_layer_norm(self.memory_adapter(message_repr))

        # Each candidate reads the message, then the candidates read each other,
        #     once per block. Read out through `decision`; that last post-norm
        #     is an `RMSNorm`, so the candidates reach it at equal length.
        refined = self.referent_decoder(adapted, memory)
        attention = self.decision(refined).squeeze(-1)

        bilinear = self.bilinear(referents, message_repr)

        weight = self.mix_weight
        mixed = (1.0 - weight) * bilinear + weight * attention
        scores = self.readout(mixed) + self.mix_bias

        # `.item()` in a forward pass costs a sync and a graph break under
        #     `torch.compile`, which is on. Paid deliberately -- a metric nobody
        #     can read is how the last collapse ran unnoticed.
        with torch.no_grad():
            self.mix_alpha = weight.item()

            # Standardised here and nowhere else. Per game, both branches are
            #     zero-mean and unit-spread, so the mean of their product over
            #     candidates *is* Pearson's r, and the weighted spreads compare
            #     like with like.
            attention_hat = standardise(attention)
            bilinear_hat = standardise(bilinear)

            self.path_agreement = (bilinear_hat * attention_hat).mean().item()

            # Which branch the score is actually made of, as opposed to which
            #     one `mix_logit` asked for. The two came apart when the
            #     branches stopped arriving at unit spread.
            attention_part = weight * attention.detach().float().std()
            bilinear_part = (1.0 - weight) * bilinear.detach().float().std()
            total = attention_part + bilinear_part
            self.mix_share = (
                (attention_part / total).item() if total > 1e-6
                else float("nan")
            )

        detached = scores.detach().float()
        spread = detached.std()
        self.decision_spread = spread.item()

        # Guarded because the fourth standardised moment divides by `spread` to
        #     the fourth, and a collapsed readout makes that 0/0. NaN is the
        #     honest value there; `decision_spread` names that state.
        if spread > 1e-6:
            standardised = (detached - detached.mean()) / spread
            self.decision_kurtosis = (standardised ** 4).mean().item() - 3.0
        else:
            self.decision_kurtosis = float("nan")

        # Note `train.py` reads `lis_scores > 0`, so accuracy is invariant to
        #     any positive rescale of the readout, which is why the accuracy
        #     column cannot see a volume collapse. `score_scale` can.
        #
        # The readout's centring moves that threshold onto each game's own mean
        #     score, and `mix_bias` is what moves it back off. The centring
        #     fixes location and scale but not shape, so an arbitrary target
        #     count stays representable; see `standardise`.
        return scores

    def reset_parameters(self):
        self.referent_adapter.reset_parameters()
        self.referent_layer_norm.reset_parameters()
        self.memory_adapter.reset_parameters()
        self.memory_layer_norm.reset_parameters()
        self.referent_decoder.reset_parameters()
        self.decision.reset_parameters()
        self.bilinear.reset_parameters()
        self.reset_score_scale()
        nn.init.constant_(self.mix_logit, float(self.mix_logit_init))
        nn.init.zeros_(self.mix_bias)


class Receiver(nn.Module):
    def __init__(
        self,
        feature_model,
        token_embedding_module,
        language_model,
        discriminator,
        dropout=0.1,
    ):
        """
        An agent that scores a set of candidate referents against a message.

        Args:
            feature_model: produces embeddings from referents
            language_model: encodes the message, `(batch, slots, width)`
            discriminator: scores the candidates from that encoding
            dropout: the listener's one dropout, on the referent embeddings.
                Counterpart to the speaker's `prototype_dropout`. Note there is
                no separate `vision_dropout` here, unlike `Sender`: it would
                mask this same tensor with nothing but a reshape between the
                two. See docs/architecture.md.

        The mask is element-wise over `(batch, n_objects, features)`, so it
            removes features within each candidate rather than removing whole
            candidates -- which would leak the label ordering.

        Applied once here and handed to both slots, rather than inside each,
            so a configuration cannot silently regularise twice. The message
            operand is left alone: it already arrives through the Gumbel
            channel, whose noise `sampling_tau` and `uniform_weight` calibrate,
            and a mask on top is a second and uncalibrated perturbation of a
            signal that has one.

        Note this puts the mask *before* each slot's norm, where both of the
            modules it replaces put it after. A `LayerNorm` following dropout
            renormalises the corrupted vector, so the two are genuinely
            different operations and the pre-split numbers are reproducible
            only at `dropout = 0`. See docs/anecdotes.md.
        """
        super().__init__()
        self.feature_model = feature_model
        self.token_embedding = token_embedding_module
        self.language_model = language_model
        self.discriminator = discriminator
        self.input_dropout = nn.Dropout(p=dropout)

    def forward(self, referents, messages):
        batch_size = referents.shape[0]
        n_obj = referents.shape[1]
        rest = referents.shape[2:]

        # Embed the referents
        referents_flat = referents.view(batch_size * n_obj, *rest)
        embedded_referents = self.feature_model(referents_flat)
        embedded_referents = embedded_referents.view(batch_size, n_obj, -1)
        embedded_referents = self.input_dropout(embedded_referents)

        # Embed the messages
        messages = messages @ self.token_embedding.weight

        message_repr = self.language_model(messages, embedded_referents)

        return self.discriminator(embedded_referents, message_repr)

    def reset_parameters(self):
        self.feature_model.reset_parameters()
        self.token_embedding.reset_parameters()
        self.language_model.reset_parameters()
        self.discriminator.reset_parameters()
