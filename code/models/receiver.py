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

    A telemetry helper, and nothing else. `AttentionDiscriminator` runs it over
        each branch separately, which is what makes `path_agreement` a Pearson r
        and lets `mix_share` compare two unstandardised branches like with like.

    It was briefly on the live path, as `ScoreVolume.readout`'s shape half. That
        is what put each game's own margin in the denominator of the gradient
        going back upstream -- see `ScoreVolume` for what it cost. Measuring a
        correlation with it is fine; deciding with it is not, and the difference
        is that the telemetry block runs under `no_grad`.

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
    The listener's one degree of freedom over how loudly it states a conclusion.
        The counterpart of the speaker's `GumbelChannel.logit_scale`: both are
        lone learned scalars in front of a normalised quantity, both go through
        `model_util.scale_without_attenuating`, and both take the same rate.
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
        `BilinearDiscriminator` layer-norms both operands of its bilinear form,
        so its score opens at `1 / sqrt(3)` at any width and any backbone by
        construction, and there is nothing left for a normaliser downstream to
        fix.

    **Why there is an offset at all.** `train.py` decides on `lis_scores > 0`,
        so the threshold is a fixed origin and the listener has to place its
        scores against it. Before this pair of parameters lived together only
        `AttentionDiscriminator` could: it had a `mix_bias`, and rungs 1-12 --
        every rung on the bilinear arm -- had no bias anywhere, `bilinear`
        being built `bias=False` and the readout a bare multiply. The bilinear
        score for candidate `j` is `LN(r_j) . proj`, so the only way to move
        all candidates together was for `proj` to align with whatever direction
        the candidates have in common, which is data-dependent and spends
        discriminative capacity in that direction. `mix_bias` is retired into
        this one: two constants across candidates are one degree of freedom
        split across two parameters.

    **Why downstream of the volume**, which is the whole reason `mix_bias` sat
        where it did. An offset applied before the scale is multiplied by it, so
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

        It is redundant. Both of `BilinearDiscriminator`'s operands are already
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
        `bilinear.weight` back out. Inside `AttentionDiscriminator` the branch
        magnitudes also set the mix, so there the norms are doubly
        load-bearing.
    """

    def _init_score_volume(self, learns_score_scale=True):
        """
        Call from `__init__` where the parameter should be created: creation
            order fixes which RNG draw every later parameter gets.

        `learns_score_scale=False` is passed only from inside
            `AttentionDiscriminator`, to the bilinear path it composes, because
            a volume there is degenerate with the one that module already has:
            the composed path is one of two branches multiplied by
            `1 - mix_weight` and then read out through this mixin downstream, so
            a scale on the branch and a move in `mix_logit` express the same
            thing and the pair would drift against each other. One volume per
            discriminator. Absent rather than frozen, so
            `split_out_parameter`'s suffix match sees the truth.

        It gates the offset too, for the matching reason: the outer readout is
            what reaches the decision, and an inner constant is annihilated by
            nothing -- it would simply be degenerate with the outer one.
        """
        self.learns_score_scale = learns_score_scale

        if self.learns_score_scale:
            # Stored as its log so `exp` keeps it strictly positive: gradient
            #     descent cannot walk a volume through zero and out the far
            #     side. Opens at 1.0, which on the bilinear arm leaves the score
            #     at its own calibrated opening of `1 / sqrt(3)` -- see
            #     `BilinearDiscriminator.forward`.
            self.log_score_scale = nn.Parameter(torch.zeros(()))

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

        A discriminator built with `learns_score_scale=False` returns the
            comparison untouched. Its caller owns both scalars for the whole
            module and a second pair here would be degenerate with them.

        **The volume goes on through `scale_without_attenuating`**, so the
            forward value is `score_scale * scores + score_bias` as it reads,
            but `d/dscores` is 1 rather than `score_scale`. The volume still
            learns, still slides, and the listener is as able to go quiet as it
            was -- what changes is only that its slide stops multiplying down
            the gradients behind it. See that function for why this is round
            nine of the same idea and for the one thing round seven's argument
            does not cover.
        """
        if not self.learns_score_scale:
            return scores

        return (
            model_util.scale_without_attenuating(scores, self.score_scale)
            + self.score_bias
        )

    def reset_score_volume(self):
        """
        Put the volume back to its 1.0 opening and the offset back to zero, so
            a reset does not leave a trained confidence or a trained threshold
            behind a fresh listener.
        """
        if self.learns_score_scale:
            with torch.no_grad():
                self.log_score_scale.zero_()
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

        # Both operands normalised, and both halves of each norm load-bearing
        #     now that nothing downstream normalises the score. The message norm
        #     divides out whatever magnitude the language model happens to emit,
        #     which would otherwise be a common factor on every candidate; the
        #     referent norm treats each candidate separately, so it changes
        #     their relative order and not just a common scale -- without it a
        #     large referent is read loudly for being large.
        projected = self.bilinear(self.message_layer_norm(message_embeddings))
        referents = self.referent_layer_norm(referents)

        scores = torch.einsum("ijh,ih->ij", (referents, projected)) # (batch, n_objects)

        # The calibration, and it is exact rather than approximate because both
        #     operands arrive normalised. Each is at per-element unit variance,
        #     so `|r| = |m| = sqrt(d)`, and with `nn.Linear`'s default init --
        #     uniform on `+/- 1/sqrt(d)`, standard deviation `1/sqrt(3d)` -- the
        #     score's standard deviation at init is `sigma_w * d = sqrt(d / 3)`.
        #     Dividing by `sqrt(d)` leaves `1/sqrt(3)` = 0.577 at every width
        #     and under every backbone, which is what makes the opening a number
        #     this repo can state rather than measure per rung.
        #
        # `log_score_scale` opens at 0, so the readout opens there too. BCE on
        #     a random map at that spread is 0.725 against `ln 2` = 0.693; at
        #     unit spread it would be 0.804. The gentler opening is the point of
        #     leaving the scalar at 1.0 rather than calibrating it to `sqrt(3)`.
        return self.readout(scores / math.sqrt(self.referent_embedding_size))

    def reset_parameters(self):
        self.bilinear.reset_parameters()
        self.reset_score_volume()
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

        Neither branch is standardised, so `a` is a weight and not a share:
            a loud branch can dominate a heavily-weighted quiet one. That is
            deliberate. Standardising per branch would make `a` mean composition
            exactly and close off the escape of turning one branch down, which
            is the failure `mix_share` against `mix_alpha` exists to catch --
            the first is the share the score is actually made of, the second the
            share `mix_logit` asked for.

        Neither is this the change that once stopped four rungs learning. That
            was the attention readout pinned to a *fixed gain*, which cannot go
            quiet while the message is still noise and so forces a listener to
            commit before there is anything to commit to (see
            docs/anecdotes.md). This listener can go as quiet as BCE asks it to,
            and the bilinear path carries the decision through the opening, so
            nothing has to be confident early.

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
        #     once, restored, and removed again when the readout's centring
        #     would have annihilated it. The centring has since gone and the
        #     bias stays off, for a third reason: it adds the same constant to
        #     every candidate, and so does `ScoreVolume.score_bias`, so the two
        #     would be degenerate and free to drift against each other. One
        #     offset per module, and it is the one on the readout, where every
        #     discriminator has it and `train.py` logs it. This is where
        #     `mix_bias` used to be named. See docs/anecdotes.md.
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
        # Built without a volume: this branch is multiplied by `1 - mix_weight`
        #     and read out through this module's own `score_scale`, so a scalar
        #     here would say the same thing `mix_logit` already says. See
        #     `ScoreVolume._init_score_volume`.
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

        # No offset here. It was `mix_bias`, a scalar added after the readout,
        #     and it is now `ScoreVolume.score_bias` -- the same scalar, the
        #     same position, applied by the same `readout`, but on both
        #     discriminators rather than only this one. `BilinearDiscriminator`
        #     had no bias anywhere, so rungs 1-12 could not place their scores
        #     against `train.py`'s fixed `lis_scores > 0` threshold at all. The
        #     argument for the position -- downstream of the volume, so the
        #     threshold does not slide every time the listener changes how
        #     loudly it speaks -- is in `ScoreVolume`'s docstring, where it now
        #     belongs. `mix_bias` also had no config key and no metrics column;
        #     `score_bias` has both.
        #
        # This is also where a `log_mix_scale` used to sit, carrying the volume
        #     for the whole module. There is again exactly one volume scalar
        #     here, but it is `ScoreVolume.log_score_scale` -- the same one the
        #     bilinear arm has, under the same config key, applied the same way.

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
        #     mean if `forward` standardised them separately.
        self.mix_share = float("nan")

        # `corr(attention_hat, bilinear_hat)` within a game. Necessary because
        #     an attention path that is never used and one that has learned to
        #     imitate the bilinear path look identical from accuracy and from
        #     `mix_alpha` alone, and they are different findings.
        self.path_agreement = float("nan")

        # The standard deviation of the scores, and independent again now that
        #     the readout does not pin it: it is `score_scale` times the mixed
        #     branches' own spread, so it reads the volume and what the module
        #     is actually doing with it together, where `score_scale` alone
        #     reads only the parameter.
        self.decision_spread = float("nan")

        # Excess kurtosis of the scores. Negative means bimodal, which is what
        #     discriminating looks like; sustained positive alongside chance
        #     accuracy is a listener with nothing to say.
        #
        # Kurtosis is invariant to the scale, so this reads shape where
        #     `decision_spread` reads magnitude. The two were briefly redundant,
        #     while the readout pinned the spread and left only this free.
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
        The two paths are mixed at their own magnitudes and the mix is scaled,
            not standardised. A branch can therefore escape being learned by
            turning itself down -- watch `mix_share` against `mix_alpha`, which
            is the only place `standardise` still runs, under `no_grad` in the
            telemetry block below where `path_agreement` and `mix_share` need
            it.

        Note this module's opening is not the bilinear arm's calibrated
            `1/sqrt(3)`: the attention branch arrives at whatever magnitude
            `decision` gives it and the mix is a weighted sum of the two. It is
            a fixed number per architecture rather than a moving one -- measure
            it with a forward pass if a rung needs its openings matched.
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
        scores = self.readout(mixed)

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
        # The threshold is a fixed zero again, not each game's own mean score:
        #     the readout no longer centres, so `score_bias` moves the threshold
        #     against a fixed origin rather than against a moving one. `train_acc`
        #     is not comparable across this change in either direction.
        return scores

    def reset_parameters(self):
        self.referent_adapter.reset_parameters()
        self.referent_layer_norm.reset_parameters()
        self.memory_adapter.reset_parameters()
        self.memory_layer_norm.reset_parameters()
        self.referent_decoder.reset_parameters()
        self.decision.reset_parameters()
        self.bilinear.reset_parameters()
        self.reset_score_volume()
        nn.init.constant_(self.mix_logit, float(self.mix_logit_init))


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
            channel, whose noise `logit_scale` and `uniform_weight` calibrate,
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
