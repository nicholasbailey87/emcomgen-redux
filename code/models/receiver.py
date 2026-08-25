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

import math

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

    Used by `AttentionDiscriminator` to put its two paths on the same footing
        before mixing them, so the mixing weight means *composition* and the
        scale downstream of it means *volume* -- the same shape/volume split
        the speaker has in `logit_spread` and `logit_scale`.

    `unbiased=False` because this is a population statistic over the candidate
        set, not an estimate from a sample of it, and the clamp keeps a game
        whose candidates happen to score identically from dividing by zero. A
        clamp is safe here in a way it is not on a parameter: nothing learns
        through this bound.
    """
    centred = scores - scores.mean(1, keepdim=True)
    spread = centred.std(dim=1, keepdim=True, unbiased=False)
    return centred / spread.clamp(min=1e-6)


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

class BilinearDiscriminator(nn.Module):
    def __init__(
        self,
        referent_embedding_size,
        message_width,
        score_scale=True,
        **kwargs
    ):
        """
        Score a candidate by a bilinear form: `score = obj_emb.T @ W @ m_emb`.

        The projection has no bias, so the score depends only on the
            relationship between message and object; a bias would add a
            message-independent per-object prior. See docs/architecture.md.

        Args:
            referent_embedding_size: width of the backbone's output
            message_width: the language model's `output_size`
            score_scale: build the learnable volume. False only from inside
                `AttentionDiscriminator`, where `standardise` runs on this
                module's output and divides any positive scale straight back
                out -- `standardise(s * u) == standardise(u)` exactly -- so the
                parameter would receive identically zero gradient, report a
                constant 1.0 in `train_score_scale`, and sit in an elevated
                learning-rate group doing nothing. The volume of a mixed score
                is `AttentionDiscriminator.log_mix_scale`, downstream of the
                standardising, and there is exactly one of it.
        """
        super().__init__()
        self.referent_embedding_size = referent_embedding_size
        self.message_width = message_width
        self.learns_score_scale = score_scale

        self.bilinear = nn.Linear(
            self.message_width,
            self.referent_embedding_size,
            bias=False
        )

        # The two operands of the dot product, normalised per example over the
        #     feature axis so the score's magnitude is not inherited from the
        #     vision model. Both are in referent space, and neither norm is
        #     affine. See docs/architecture.md.
        self.referent_layer_norm = nn.LayerNorm(
            self.referent_embedding_size,
            elementwise_affine=False,
            eps=LAYER_NORM_EPS,
        )
        self.message_layer_norm = nn.LayerNorm(
            self.referent_embedding_size,
            elementwise_affine=False,
            eps=LAYER_NORM_EPS,
        )

        # The listener's one degree of freedom over its own confidence, and the
        #     counterpart of the speaker's `log_logit_scale`. One scalar, opening
        #     at 1.0, stored as a log. Unbounded deliberately: the arm that
        #     learns lets this *fall* 0.856 -> 0.238 and still reaches 1.000, so
        #     a floor would be solving a non-problem. See docs/architecture.md.
        #
        # Absent rather than frozen when `score_scale` is False, so that
        #     `split_out_parameter`'s suffix match and `train.py`'s dispatch
        #     both see the truth instead of a parameter that cannot move.
        if self.learns_score_scale:
            self.log_score_scale = nn.Parameter(torch.zeros(()))

    @property
    def score_scale(self):
        """
        The multiplier applied to the normalised message operand, always
            positive. Read here rather than exponentiating at the use site, so
            `forward` and the metrics column cannot drift apart.
        """
        if not self.learns_score_scale:
            return torch.ones((), device=self.bilinear.weight.device)
        return self.log_score_scale.exp()

    def forward(
        self,
        referents: torch.Tensor, # (batch, n_objects, d_embedding)
        message_repr: torch.Tensor # (batch, slots, message_width)
        ) -> torch.Tensor: # -> (batch, n_objects)
        """
        The mean over slots is the identity for `ReceiverGRULM`, which returns
            one. For a stack that returns one position per symbol it is the
            honest analogue of "the last state", which a bidirectional encoder
            does not have.
        """
        message_embeddings = message_repr.mean(1)

        # Normalised *after* the projection: `bilinear` is free, so a norm on
        #     its input would set only where `W` starts.
        projected = self.bilinear(message_embeddings)
        projected = self.message_layer_norm(projected) * self.score_scale

        referents = self.referent_layer_norm(referents)

        scores = torch.einsum("ijh,ih->ij", (referents, projected)) # (batch, n_objects)

        # Attention's `1/sqrt(d)`, and load-bearing now that both operands are
        #     normalised: it is what makes `score_scale = 1.0` the calibrated
        #     opening at any `referent_embedding_size`.
        return scores / math.sqrt(self.referent_embedding_size)

    def reset_parameters(self):
        self.bilinear.reset_parameters()
        # No-ops while the two norms are parameter-free, and listed anyway so
        #     that turning `elementwise_affine` back on cannot leave a reset
        #     listener holding trained gains.
        self.referent_layer_norm.reset_parameters()
        self.message_layer_norm.reset_parameters()
        if self.learns_score_scale:
            nn.init.zeros_(self.log_score_scale)


class AttentionDiscriminator(nn.Module):
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

            score = s * [ (1 - a) * bilinear_hat + a * attention_hat ] + bias
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

        Both paths are standardised per game before mixing, so `a` means
            composition and `s` alone means volume.

        Note this standardises the attention readout, which is the thing that
            once stopped four rungs learning: a readout pinned to a fixed gain
            cannot go quiet while the message is still noise, and a listener
            forced to commit from step zero commits before there is anything to
            commit to. See docs/anecdotes.md. What makes it survivable here is
            the bilinear path -- it carries the decision through the opening,
            so nothing has to be confident early -- and `log_mix_scale`, which
            leaves the *pair* free to go quiet even though neither path can do
            it alone. What is closed is only the escape of turning the
            attention path down to avoid learning it.

        The floor is in the parameterisation and **never a `clamp`**:
            `clamp`'s gradient is zero below the bound, so a mix that drifted
            under the floor would weld there permanently. With the floor in
            place the attention path always contributes and so always receives
            gradient, and because standardising it means it cannot go quiet,
            an uninformative attention path cannot be escaped by turning it
            down -- it has to be made informative or paid for.

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

        # A bias, and a weight whose magnitude is free. Both were removed once
        #     and restored; see docs/anecdotes.md. It reads straight off the
        #     referent stack's last post-norm, which is an `RMSNorm` and so
        #     already equalises the candidates' lengths -- no object can be read
        #     loudly for being large.
        self.decision = nn.Linear(self.d_model, 1)

        # The other path. A whole `BilinearDiscriminator`, composed rather than
        #     reimplemented, so the `a -> mix_floor` limit of this module is
        #     the module that was measured bootstrapping and not a lookalike.
        #     It reads `message_repr`; it owns no encoder.
        self.bilinear = BilinearDiscriminator(
            self.referent_embedding_size, self.message_width, score_scale=False
        )

        # `mix_logit_init` -4.0 puts `a` at 0.116 for the default floor of
        #     0.1 -- open essentially at the bilinear comparison, with the
        #     attention path present enough to be learning.
        self.mix_logit = nn.Parameter(
            torch.tensor(float(self.mix_logit_init))
        )

        # Volume, downstream of the mix. Log-parameterised, unbounded and
        #     opening at 1.0, exactly as `log_score_scale` is. Both operands are
        #     standardised, so nothing upstream can absorb it.
        self.log_mix_scale = nn.Parameter(torch.zeros(()))
        self.mix_bias = nn.Parameter(torch.zeros(()))

        # Metrics only: set on every `forward`, read by `train.py`. See
        #     docs/measurement.md.
        #
        # How much of the score is the attention path. Bounded below by
        #     `mix_floor` and above by 1, so it reads directly as "was
        #     attention used".
        self.mix_alpha = float("nan")

        # The volume, and this arm's counterpart of `score_scale`. The two
        #     mixed paths are standardised, so this scalar is the only thing
        #     that sets the score's magnitude -- read it the way
        #     `train_score_scale` is read on the bilinear arm.
        self.recorded_mix_scale = float("nan")

        # `corr(attention_hat, bilinear_hat)` within a game. Necessary because
        #     an attention path that is never used and one that has learned to
        #     imitate the bilinear path look identical from accuracy and from
        #     `mix_alpha` alone, and they are different findings.
        self.path_agreement = float("nan")

        # The standard deviation of the scores. A monotone descent towards zero
        #     is the finding, not wandering.
        self.decision_spread = float("nan")

        # Excess kurtosis of the scores. Negative means bimodal, which is what
        #     discriminating looks like; sustained positive alongside chance
        #     accuracy is a listener with nothing to say.
        self.decision_kurtosis = float("nan")

    @property
    def mix_weight(self):
        """
        The attention path's share of the score, in `[mix_floor, 1)`. Read
            here rather than recomputed at the use site so `forward` and the
            metrics column cannot drift apart.
        """
        return (
            self.mix_floor
            + (1.0 - self.mix_floor) * torch.sigmoid(self.mix_logit)
        )

    @property
    def mix_scale(self):
        return self.log_mix_scale.exp()

    def forward(
        self,
        referents: torch.Tensor, # (batch, n_objects, d_embedding)
        message_repr: torch.Tensor # (batch, slots, message_width)
        ) -> torch.Tensor: # -> (batch, n_objects)
        adapted = self.referent_layer_norm(self.referent_adapter(referents))
        memory = self.memory_layer_norm(self.memory_adapter(message_repr))

        # Each candidate reads the message, then the candidates read each other,
        #     once per block. Read out through `decision`; that last post-norm
        #     is an `RMSNorm`, so the candidates reach it at equal length.
        refined = self.referent_decoder(adapted, memory)
        attention_hat = standardise(self.decision(refined).squeeze(-1))

        bilinear_hat = standardise(self.bilinear(referents, message_repr))

        weight = self.mix_weight
        mixed = (1.0 - weight) * bilinear_hat + weight * attention_hat
        scores = self.mix_scale * mixed + self.mix_bias

        # `.item()` in a forward pass costs a sync and a graph break under
        #     `torch.compile`, which is on. Paid deliberately -- a metric nobody
        #     can read is how the last collapse ran unnoticed.
        with torch.no_grad():
            self.mix_alpha = weight.item()
            self.recorded_mix_scale = self.mix_scale.item()
            # Both operands are already per-game zero-mean and unit-spread, so
            #     the mean of their product over candidates *is* Pearson's r.
            self.path_agreement = (bilinear_hat * attention_hat).mean().item()

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
        #     column cannot see a volume collapse. `decision_spread` can.
        return scores

    def reset_parameters(self):
        self.referent_adapter.reset_parameters()
        self.referent_layer_norm.reset_parameters()
        self.memory_adapter.reset_parameters()
        self.memory_layer_norm.reset_parameters()
        self.referent_decoder.reset_parameters()
        self.decision.reset_parameters()
        self.bilinear.reset_parameters()
        nn.init.constant_(self.mix_logit, float(self.mix_logit_init))
        nn.init.zeros_(self.log_mix_scale)
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
