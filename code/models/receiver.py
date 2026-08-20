"""
Listener models
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import broccoli

from . import model_util

# Mirrors `sender.LAYER_NORM_EPS`, and load-bearing for the same reason.
#     `F.layer_norm` divides by `sqrt(var + eps)`, so scale invariance holds only
#     while the incoming variance is large against `eps`; below that the
#     normaliser quietly stops normalising and the operand comes through smaller
#     than unit variance, taking the score's magnitude back out of
#     `score_scale`'s hands and putting it back in the backbone's. At the 1e-5
#     default a referent at RMS 0.01 comes out 4.5% off; at 1e-12 it is exact to
#     four decimal places. Not currently binding -- ViT2 emits RMS 0.23 -- but it
#     is the same trap `layer_norm_logits` documents, and closing it costs
#     nothing. See `sender.layer_norm_logits` for the derivation.
LAYER_NORM_EPS = 1e-12

# Every broccoli module below is constructed with its full argument list, even
#     where an argument is being set to the value it would have defaulted to.
#     broccoli's defaults are not a stable interface: between 27.1.1 and 30.1.0
#     `TransformerEncoder` flipped from `pre_norm=True, post_norm=False` to the
#     reverse, which would have silently inverted the architecture of every
#     model here on a `pip install`, with no error and no diff in this repo.
#     Arguments that are inert under the current settings are still set, so
#     that a future default change cannot quietly make them live.


class BilinearGRUComparer(nn.Module):
    def __init__(
        self,
        referent_embedding_size,
        **kwargs
    ):
        """
        Use a bilinear model to compare embedded messages with sets of possible
            message referents.

        This model uses a linear layer (self.bilinear) to project the message
            embedding (`m_emb`) from `message_size` to `feature_size`, allowing
            us to take a dot product between message and referent embeddings.

        Bias in the projection layer is set to False. This is a deliberate
            choice to create a "pure" bilinear interaction, where the score is
            based *only* on the relationship between the message and the object.

        The calculation for a single object/message pair is:

        1. Project message:  m_emb_projected = weights @ m_emb
        2. Compute score:   score = obj_emb.T @ m_emb_projected

        This results in the pure bilinear form:

            score = obj_emb.T @ weights @ m_emb

        ---
        Why not use a bias?
        If bias were True, the calculation would be:

        1. Project message:
            (weights @ m_emb) + bias
        2. Compute score:
            score = obj_emb.T @ m_emb_projected

        This expands to:

            (obj_emb.T @ weights @ m_emb) + (obj_emb.T @ bias)

        That second term, (obj_emb.T @ bias), is a message-independent "prior"
            that would make the model prefer certain objects regardless of what
            the message said. We disable the bias to prevent this.
        """
        super().__init__()
        self.referent_embedding_size = referent_embedding_size
        self.token_embedding_size = kwargs["token_embedding_size"]
        self.d_model = kwargs["d_model"]
        # `dropout` means exactly one thing across both comparers: mask the
        #     incoming referent embeddings, after their norm and immediately
        #     before the comparison, where there is no averaging left
        #     downstream to restore the masked units. It is the listener's only
        #     regulariser and the counterpart of the sender's
        #     `prototype_dropout`. Module internals are fixed constants below,
        #     so raising this knob never silently rewires the architecture.
        #
        # It used to mask the message operand too, and the argument for that
        #     was that a dot product lets the listener lean on whichever side
        #     is left intact. True, but it assumed the two sides arrive on
        #     equal terms and they do not: the message comes through the Gumbel
        #     channel, whose noise is already calibrated by `sampling_tau` and
        #     `uniform_weight`, so a mask on top is a second perturbation of a
        #     signal that has one -- and the listener cannot tell which of the
        #     two it is being asked to be robust to. The referents arrive clean.
        self.dropout = nn.Dropout(p=kwargs["dropout"])
        self.bidirectional = kwargs["bidirectional"]
        self.layers = kwargs["layers"]

        self.gru = nn.GRU(
            self.token_embedding_size,
            self.d_model,
            num_layers=self.layers,
            bias=True,
            batch_first=True,
            # Fixed at 0.0 to match jayelm, whose listener GRU takes no dropout
            #     argument at all (`rnn.py:21`). Inert either way while
            #     `layers = 1` — PyTorch only applies this *between* layers —
            #     but wiring the knob in here meant it would switch on
            #     unannounced the moment anyone raised `layers`. Zero also
            #     silences PyTorch's warning about that combination.
            dropout=0.0,
            bidirectional=self.bidirectional
        )

        gru_output_dim = (
            self.d_model * 2
            if self.bidirectional
            else self.d_model
        )

        self.bilinear = nn.Linear(
            gru_output_dim,
            self.referent_embedding_size,
            bias=False
        )

        # The two operands of the dot product, normalised per example over the
        #     feature axis so the score's magnitude stops being inherited from
        #     whichever vision model the rung mounts. Both are in referent space:
        #     the message operand is `bilinear`'s *output*, not the GRU state,
        #     because a norm upstream of a free `Linear` constrains nothing
        #     downstream of it.
        #
        # No affine on either. The score is `r . p`, so a per-dimension gain is
        #     absorbable into `bilinear` and could only add a second, unbounded
        #     route to score magnitude -- the one these exist to close. It also
        #     keeps `sum(LN(r)) = 0`, which is what annihilates the message
        #     operand's mean-subtraction; with a `beta` that term would start
        #     shifting scores between objects.
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
        #     counterpart of the speaker's `log_logit_scale`. Normalising both
        #     operands closes every other route to score magnitude, and BCE is
        #     not scale-invariant, so without this the listener could only ever
        #     sharpen by aligning the two -- never by committing harder to an
        #     alignment it already has.
        #
        # One scalar, not one per operand: `c * LN(p) . LN(r)` and
        #     `LN(p) . c * LN(r)` are the same function, so a second would be
        #     redundant with only the product able to act. It multiplies the
        #     message operand, which is shared across the objects of a game, so
        #     it cannot change which object wins -- only how loudly the listener
        #     says so.
        #
        # Opens at 1.0, which `forward`'s `1/sqrt(d)` makes the calibrated
        #     value: both operands leave LayerNorm at norm `sqrt(d)` and start
        #     mutually random, so the division puts the untrained score at unit
        #     standard deviation and BCE within a hair of `ln 2` on both arms.
        #     Nothing here has a traverse to cover -- unlike
        #     `log_logit_scale`, which opens at 0.839 against a usable channel
        #     of 4 to 6.
        #
        # Stored as a log anyway, for reasons that are not that one. Zero is
        #     where every gradient in the pair is gated, since `s` multiplies
        #     the only path from the message to the loss, and `exp` puts it out
        #     of reach; halving and doubling a gain should cost the same step;
        #     and it gives `train_score_scale` a known ceiling of
        #     `score_scale_lr * steps` log-units per epoch, which is what makes
        #     the column readable rather than merely present.
        self.log_score_scale = nn.Parameter(torch.zeros(()))

    @property
    def score_scale(self):
        """
        The multiplier applied to the normalised message operand, always
            positive. Read here rather than exponentiating at the use site, so
            `forward` and the metrics column cannot drift apart. Counterpart of
            `SenderGRULM.logit_scale`.
        """
        return self.log_score_scale.exp()

    def forward(
        self,
        referents: torch.Tensor, # (batch, n_objects, d_embedding)
        messages: torch.Tensor # (batch, seq_len, d_embedding)
        ) -> torch.Tensor: # -> (batch, n_objects)
        """
        Takes a batch of sets of embedded referents, of shape
            (batch_size, n_obj, referent_embedding_size)
            and a batch of messages with embedded tokens, of shape
            (batch_size, message_length, message_embedding_size)

        Returns a batch of scores, of shape (batch_size, n_obj)
        """
        token_embeddings, _ = self.gru(messages) # (b, seq, directions * d_model)

        # Taking timestep -1 gives the state after the GRU has consumed the
        #     last *slot*, not the last real token. That is correct here only
        #     because our messages are never padded: `mask_reserved_tokens`
        #     puts PAD/SOS/EOS/UNK at -inf so the sender cannot emit EOS
        #     mid-message, and `SenderGRULM.decode` always builds
        #     SOS + (message_length - 2) content symbols + EOS. Every message
        #     is therefore exactly `message_length` long and position -1 is
        #     always the real EOS.
        #
        # This diverges from jayelm, whose speaker *does* sample EOS early and
        #     tracks a per-example `lang_length`, leaving shorter messages
        #     padded at the end; their listener has to `pack_padded_sequence`
        #     to avoid exactly the failure this comment used to warn about.
        #
        # So the assumption is dormant, not satisfied by design elsewhere. It
        #     breaks the moment either of these changes:
        #       - EOS is dropped from the reserved mask to let the sender
        #         choose message length (the more faithful reproduction), at
        #         which point this reads post-EOS junk;
        #       - anything feeds padded language to the receiver, e.g. an
        #         ACRe-style eval replaying sampled messages. Nothing does
        #         today; every call site takes `lang` straight from the sender.
        #     The bidirectional branch has the mirror exposure: it reads the
        #     backward pass at position 0, which under end-padding is the state
        #     after that pass has run *through* the padding first.
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
        
        # Normalised *after* the projection, not before it: `bilinear` is free,
        #     so a norm on its input would set only where `W` starts, where a
        #     norm on its output divides `W`'s magnitude out of the score
        #     entirely and leaves `score_scale` as the one thing that sets it.
        projected = self.bilinear(message_embeddings)
        projected = self.message_layer_norm(projected) * self.score_scale

        # The referents are masked and the message is not. The message operand
        #     arrives through the Gumbel channel, which is already a noise
        #     process the listener has to be robust to and one whose scale
        #     `sampling_tau` and `uniform_weight` are calibrated to set; a
        #     dropout mask on top of it is a second, uncalibrated perturbation
        #     of the same signal, and the listener cannot tell the two apart.
        #     The referents arrive clean, so this is the only side where a mask
        #     regularises rather than compounds.
        #
        # `TransformerCrossAttentionComparer` masks the same one side, so the
        #     key still means one thing across both classes. It stopped meaning
        #     "both operands" here rather than starting to mean it there.
        referents = self.referent_layer_norm(referents)
        referents = self.dropout(referents)

        scores = torch.einsum("ijh,ih->ij", (referents, projected)) # (batch, n_objects)

        # Attention's `1/sqrt(d)`, and load-bearing rather than cosmetic now
        #     that both operands are normalised. It is what makes
        #     `score_scale = 1.0` the calibrated opening instead of a number
        #     whose meaning moves with `referent_embedding_size` -- 512 on
        #     ResNet18 against 320 on ViT2, which would otherwise open the two
        #     arms 1.26x apart and both far too loud.
        return scores / math.sqrt(self.referent_embedding_size)

    def reset_parameters(self):
        self.gru.reset_parameters()
        self.bilinear.reset_parameters()
        # No-ops while the two norms are parameter-free, and listed anyway so
        #     that turning `elementwise_affine` back on cannot leave a reset
        #     listener holding trained gains -- the failure the other comparer's
        #     `reset_parameters` already had once.
        self.referent_layer_norm.reset_parameters()
        self.message_layer_norm.reset_parameters()
        nn.init.zeros_(self.log_score_scale)


class TransformerCrossAttentionComparer(nn.Module):
    def __init__(
        self,
        referent_embedding_size,
        **kwargs
    ):
        """
        Compare an embedded message against a set of candidate referents by
            reading each against the other, twice.

        The message reads the candidate set first (`message_cross_attention`),
            so that `encoding` refines a meaning that already knows what it is
            choosing between; then each candidate reads the refined message
            (`referent_cross_attention`); then the candidates read each other
            (`referent_self_attention`); then a normalised linear readout
            scores each one.

        ---
        Why the message reads the referents before it is encoded

        Without that first pass the encoder sees the message alone, so the best
            it can build is an *absolute* meaning -- "a red square" -- when the
            task is discriminative and what distinguishes the target from this
            particular set of distractors may be something else entirely. The
            candidate set is not privileged information: the listener is
            holding it. Letting the message query it is the difference between
            encoding what the message says and encoding what the message says
            *about these objects*.

        This costs the first cross-attention its position information, because
            `encoding` is where position is embedded and it now runs second. So
            two identical symbols in different slots query the candidate set
            identically, and `encoding` has to tell them apart from context
            afterwards. Cheap at `message_length` 7 to 10, and the alternative
            -- lifting absolute position out of broccoli's encoder into this
            class -- buys little for the wiring it costs.

        ---
        Why every residual is post-normed rather than a bare add

        `MHAttention` already RMS-normalises its output, so `x + attn(...)`
            adds two tensors of norm `sqrt(d)` and the residual stream grows by
            `sqrt(2)` per stage -- 2.8x across the three here. Each add is
            therefore `RMSNorm(alpha * x + beta * attended)`, which is what
            broccoli's `EncoderBlock` does internally and what DeepNorm's
            constants are derived for.

        ---
        Why the readout is normalised and the scale is a separate scalar

        `decision` used to be a bare `nn.Linear(d_model, 1)`, which made one
            vector both the *direction* the head reads out and the *volume* it
            reads out at. BCE will always reduce a loss it cannot otherwise
            reduce by becoming less confident, and that pressure is
            first-order where learning a useful direction is not, so the volume
            collapses first. On CUB it did exactly that: scores fell from sd
            0.42 to sd 0.016 inside one epoch and stayed there for thirty,
            with `train_loss` pinned at `ln 2 + 2e-5`.

        That is worse than a bad score, because every gradient reaching
            `referent_self_attention`, both cross-attentions, `encoding`, both
            adapters, both vision models and the entire speaker is proportional
            to the readout's magnitude. A quiet listener starves the machinery
            that would make it informative, which is why thirty epochs did not
            escape it.

        So the direction is normalised to a unit vector in `forward` -- which
            leaves its gradient orthogonal to itself, making the magnitude
            inert -- and the magnitude lives in `log_score_scale`, exactly as
            `bilinear`'s magnitude is divided out by `message_layer_norm` and
            replaced by `BilinearGRUComparer.log_score_scale`. Same defect,
            same fix, one module later.
        """
        super().__init__()
        self.d_model = kwargs["d_model"]
        self.token_embedding_size = kwargs["token_embedding_size"]
        self.referent_embedding_size = referent_embedding_size
        self.message_length = kwargs["message_length"]
        self.dropout = kwargs["dropout"]
        self.layers = kwargs["layers"]
        self.heads = kwargs["heads"]
        self.utility_tokens = kwargs["utility_tokens"]
        self.bidirectional = kwargs["bidirectional"]
        self.ff_inner_size = kwargs["ff_inner_size"]
        self.cross_attention_dropout = kwargs["cross_attention_dropout"]
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

        # `layers` is the message encoder's depth, and nothing else's. It used
        #     to be a total split between a reading stack and a fusion stack,
        #     which meant `layers = 5` bought a 3-layer encoder and asking for
        #     one more block moved two.
        self.encoding_alpha, self.encoding_beta = (
            model_util.resolve_residual_scaling(
                kwargs["alpha"], kwargs["beta"], self.layers
            )
        )

        # One pair for the three hand-written residuals, resolved at depth 1.
        #     DeepNorm's depth argument counts attention-plus-feedforward
        #     *layers*, and these are bare attention sublayers: one on the
        #     message stream, two on the referent stream. A stream of two
        #     attention sublayers and no feedforward is one layer's worth of
        #     residual path, so 1 is the honest depth for both. Pinning a
        #     number in config still passes straight through, as everywhere.
        self.residual_alpha, self.residual_beta = (
            model_util.resolve_residual_scaling(
                kwargs["alpha"], kwargs["beta"], 1
            )
        )

        # Suppressed unless the encoder is deep enough for a depth ramp to mean
        #     anything: `depthwise_linear_stochastic_depth` spreads the rate
        #     linearly across layers, so a one-layer stack would get a single
        #     rate of 0.0 regardless. The three residuals below get none --
        #     they are not `EncoderBlock`s and have no branch to drop.
        self.stochastic_depth = (
            kwargs["stochastic_depth"] if self.layers > 1 else 0.0
        )

        # The listener's regulariser, and it masks the referents only. See
        #     `BilinearGRUComparer.__init__` for why the message operand is
        #     left alone: it arrives through the Gumbel channel, whose noise is
        #     already calibrated, and a mask on top is a second perturbation of
        #     the same signal. Attention dropout is a separate setting
        #     (`receiver_comparer.cross_attention_dropout`), not this knob.
        #
        # Placed after `referent_layer_norm` rather than before the adapter,
        #     which is where it used to be. A mask upstream of a learned
        #     projection is a mask the projection can average away, and a mask
        #     upstream of a LayerNorm has its `1/(1-p)` rescale thrown away and
        #     its survivors renormalised *up*, so the perturbation is neither
        #     the size nor the shape the knob names.
        self.input_dropout = nn.Dropout(p=self.dropout)

        # `bias=False`, and load-bearing rather than tidy. `referent_layer_norm`
        #     below is what makes the score independent of the size the vision
        #     model happens to emit, and it can only do that exactly if what
        #     reaches it is homogeneous in the input: `W(cx) = cW(x)` gives
        #     `LN(W(cx)) = LN(W(x))`, where `W(cx) + b` does not. With a bias,
        #     a backbone emitting features a hundred times smaller gets a score
        #     shaped partly by this layer's bias and one emitting large features
        #     does not -- a weaker form of exactly the defect being removed, and
        #     one that would leave the invariance test asserting an
        #     approximation. The following norm subtracts the mean anyway, so
        #     most of a bias here would be annihilated a line later.
        self.referent_adapter = nn.Linear(
            self.referent_embedding_size,
            self.d_model,
            bias=False
        )

        # Parameter-free, as on `BilinearGRUComparer`. Not for the reason
        #     originally given here: broccoli's `project_qkv` RMS-normalises Q
        #     and K per head, so the attention *logits* are already free of the
        #     vision model's scale, and `MHAttention.out_norm` handles a
        #     uniformly louder backbone (measured: the whole set at 10x moves
        #     the output by 0.0%).
        #
        # What neither handles is *per-object* magnitude, and at
        #     `message_cross_attention` the referents are the values. V is not
        #     normed anywhere, so the attention output is a magnitude-weighted
        #     mixture: one candidate 50x larger than its neighbours moves that
        #     output by 116% without this norm and by 0.0% with it, and no
        #     downstream norm can undo it because the averaging has already
        #     happened. That is an object winning for being large rather than
        #     for matching, which is the failure
        #     `test_the_referent_norm_is_not_a_global_rescale` pins on the
        #     other comparer.
        #
        # An affine would also be a second route to score magnitude alongside
        #     `log_score_scale`, which is the route this class exists to close.
        self.referent_layer_norm = nn.LayerNorm(
            self.d_model, elementwise_affine=False, eps=LAYER_NORM_EPS
        )

        self.message_adapter = nn.Linear(
            self.token_embedding_size,
            self.d_model
        )

        # Stage 1: the message queries the candidate set.
        self.message_cross_attention = self._attention()
        self.message_residual_norm = nn.RMSNorm(self.d_model)

        # Stage 2: refine that reading in context.
        self.encoding = broccoli.transformer.TransformerEncoder(
            self.message_length, # seq_len can be none as length-invariant
            self.d_model,
            self.layers,
            self.heads,
            # Pinned False, and no longer a config option. Every stack here runs
            #     rotary, which encodes position where it is used -- as a
            #     rotation of the query and key subspace -- rather than as a
            #     vector added to the residual stream once at the input. A
            #     rotation of part of the vector is sufficient for relative
            #     position, so an absolute embedding on top is not covering
            #     anything RoPE leaves out; it is a second, differently-shaped
            #     answer to the same question, learned from scratch, and one
            #     that has to be re-learned for every sequence length. broccoli
            #     agrees: its own `ViT` defaults to exactly this pair, False
            #     absolute and True relative.
            #
            # It cost ~190k parameters a rung, most of it two 289-position
            #     tables in the ViT2 backbones.
            absolute_position_embedding=False,
            relative_position_embedding=self.relative_position_embedding,
            # Pinned at 1.0, not configurable -- see `ViT2` for the argument.
            positional_heads=1.0,
            # Derived from the data, not configured separately: this block
            #     reads the message, so its source is the message length.
            source_size=(self.message_length,),
            # `ff_ratio` None so that `ff_inner_size` is the live knob -- see
            #     `SenderTransformerLM`, and note broccoli's `ViT` resolves the
            #     two the other way round.
            ff_ratio=None,
            ff_inner_size=self.ff_inner_size,
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
            causal=not self.bidirectional,
            linear_module=nn.Linear,
            bos_tokens=self.utility_tokens,
            knocking_heads=self.knocking_heads,
            return_bos_tokens=self.return_bos_tokens,
            pre_norm=self.pre_norm,
            post_norm=self.post_norm,
            msa_scaling="d",
            alpha=self.encoding_alpha,
            beta=self.encoding_beta,
        )

        # Stage 3: each candidate queries the refined message.
        self.referent_cross_attention = self._attention()
        self.referent_residual_norm = nn.RMSNorm(self.d_model)

        # Stage 4: the candidates query each other, which is the only stage at
        #     which a score can depend on the rest of the set. Redundant for a
        #     criterion like "bigger than average", which the message could
        #     carry on its own; load-bearing for one like "the odd one out",
        #     which no per-object reading can express. Neither is in the task
        #     as it stands, and it is the stage this class had all along --
        #     `fusion`, minus the feedforward.
        self.referent_self_attention = self._attention()
        self.referent_self_attention_norm = nn.RMSNorm(self.d_model)

        # Parameter-free, and not optional. broccoli's post-norm is
        #     `nn.RMSNorm(d_model)` with `elementwise_affine=True` by default,
        #     so `referent_self_attention_norm` above carries a learnable gain
        #     -- a second, unbounded route to score magnitude that BCE would
        #     drive the collapse through instead. That affine cannot simply be
        #     turned off: the same class is used by `encoding` and by the
        #     speaker's stacks, where it is wanted. So the readout normalises
        #     downstream of it.
        self.decision_layer_norm = nn.LayerNorm(
            self.d_model, elementwise_affine=False, eps=LAYER_NORM_EPS
        )

        # Held as an `nn.Linear` for its initialisation and its
        #     `reset_parameters`, but deliberately never *called* -- `forward`
        #     normalises `weight` first. Calling it directly would restore the
        #     free magnitude this class was rewritten to remove, so if you find
        #     yourself writing `self.decision(...)`, that is the bug.
        self.decision = nn.Linear(self.d_model, 1, bias=True)

        # The one thing that sets how loudly this listener states a conclusion,
        #     and the exact counterpart of `BilinearGRUComparer`'s. Opens at
        #     1.0, which is calibrated rather than arbitrary: `decision_layer_norm`
        #     leaves `refined` at norm `sqrt(d)` and the readout direction is a
        #     unit vector, so their product opens at unit variance and BCE
        #     within a hair of `ln 2`.
        #
        # Stored as a log for the reasons given on the other comparer, plus one
        #     specific to this class: a linear scale does not stop at zero. At
        #     `score_scale_lr = 2e-3` AdamW moves a lone scalar about `lr` a
        #     step, so from 1.0 it would reach zero in 500 steps -- under three
        #     CUB epochs -- and then go negative, inverting the readout's sign
        #     while the loss looked unremarkable. `exp` forbids that, and bounds
        #     the descent to `score_scale_lr * steps` log-units, which is what
        #     makes the collapse this class just had visible in
        #     `train_score_scale` rather than recoverable only by algebra on the
        #     fourth decimal place of `train_loss`.
        self.log_score_scale = nn.Parameter(torch.zeros(()))

    def _attention(self):
        """
        One bare `MHAttention` at this module's width, built the same way three
            times. Bare rather than an `EncoderBlock` because the residual and
            its post-norm are written out in `forward`, where the two streams
            they join are visible.
        """
        return broccoli.transformer.MHAttention(
            self.d_model,
            self.heads,
            # Architecture internals, deliberately not `self.dropout` — that
            #     knob is the listener's regulariser on the comparison inputs,
            #     and should not double as an attention-internals setting. The
            #     speaker's cross-attention takes the identically named key, so
            #     both agents' attention is regularised on the same terms.
            #     broccoli gates it on `self.training` (the `dropout_p`
            #     argument in `MHAttention.forward`), so it does not leak into
            #     eval the way a bare
            #     `F.scaled_dot_product_attention(dropout_p=...)` would.
            dropout=self.cross_attention_dropout,
            # Inert while `causal=False`, and that is not negotiable for the
            #     two stages whose sequence axis is the referent set. In this
            #     codebase referent *order is the label vector*:
            #     `data.util.split_spk_lis` writes positives into the first
            #     half of each agent's view and negatives into the second, and
            #     the augmentation in `ConceptDataset.__getitem__` (and
            #     `CUBDataset.sample_game`) permutes only *within* each half.
            #     Anything here that could index its own sequence axis could
            #     learn "the first half are targets" and score perfectly while
            #     ignoring the message. With no position embedding and no mask,
            #     all three of these are permutation-equivariant and cannot read
            #     the ordering at all. `BilinearGRUComparer` is immune for a
            #     different reason: it scores each referent in isolation and
            #     never sees the set.
            causal=False,
            seq_len=self.message_length,
            linear_module=nn.Linear,
            bos_tokens=0,
            knocking_heads=False,
            # No positional information in any of the three. On the referent
            #     axis that is the ordering argument above. On the message axis
            #     `encoding` is where position is embedded, and stage 1 runs
            #     before it -- see the class docstring for what that costs.
            #     `positional_heads` is inert while `rotary_embedding` is None
            #     and is pinned at the repo-wide 1.0 anyway, so that turning
            #     rotary on here could not quietly introduce a head partition.
            #     Note broccoli defaults it to 0.25 on `MHAttention` and 0.5 on
            #     `TransformerEncoder`, so the two are not interchangeable.
            rotary_embedding=None,
            positional_heads=1.0,
            source_size=None,
            scaling="d",
        )

    @property
    def score_scale(self):
        """
        The multiplier applied to the normalised readout, always positive. Read
            here rather than exponentiating at the use site, so `forward` and
            the metrics column cannot drift apart. Counterpart of
            `BilinearGRUComparer.score_scale`.
        """
        return self.log_score_scale.exp()

    def _residual(self, stream, attended, norm):
        """
        DeepNorm's post-norm residual, written out because these three joins
            are between two different streams rather than inside one block.
        """
        return norm(self.residual_alpha * stream + self.residual_beta * attended)

    def forward(
        self,
        referents: torch.Tensor, # (batch, n_objects, d_embedding)
        messages: torch.Tensor # (batch, seq_len, d_embedding)
        ) -> torch.Tensor: # -> (batch, n_objects)
        """
        Takes a batch of sets of embedded referents, of shape
            (batch_size, n_obj, referent_embedding_size)
            and a batch of messages with embedded tokens, of shape
            (batch_size, message_length, message_embedding_size)

        Returns a batch of scores, of shape (batch_size, n_obj)
        """
        referents = self.referent_adapter(referents)
        referents = self.referent_layer_norm(referents)
        referents = self.input_dropout(referents)

        messages = self.message_adapter(messages)

        # 1. The message reads the candidate set, so what `encoding` refines is
        #    a discriminative meaning rather than an absolute one.
        messages = self._residual(
            messages,
            self.message_cross_attention(messages, referents, referents),
            self.message_residual_norm,
        )

        # 2. Refine it. Note this *mutates* `messages`: broccoli's
        #    `TransformerEncoder.preprocess` adds its position embedding with
        #    `x += position_embedding`, in place, on the tensor handed to it.
        #    Harmless here because nothing reads `messages` again -- but a
        #    second residual taken from the pre-encoding message would silently
        #    be reading a positional embedding as well, so take a copy first if
        #    one is ever added.
        encoded_messages = self.encoding(messages)

        # 3. Each candidate reads the refined message. The residual is what
        #    carries referent identity to the readout *linearly* -- without it
        #    a candidate reaches the score only through near-uniform attention
        #    weights, and this stage halved the between-object share of the
        #    variance (0.415 going in, 0.221 coming out) at init.
        enriched = self._residual(
            referents,
            self.referent_cross_attention(
                referents, encoded_messages, encoded_messages
            ),
            self.referent_residual_norm,
        )

        # 4. The candidates read each other.
        refined = self._residual(
            enriched,
            self.referent_self_attention(enriched, enriched, enriched),
            self.referent_self_attention_norm,
        )

        # 5. Read out along a direction of unit length, at a volume set by one
        #    scalar. `F.normalize` leaves `decision.weight`'s gradient
        #    orthogonal to itself, so its magnitude is inert and cannot become
        #    a second route to confidence.
        refined = self.decision_layer_norm(refined)
        direction = F.normalize(self.decision.weight, dim=-1) # (1, d_model)
        scores = refined @ direction.squeeze(0) # (batch, n_objects)

        # The bias sits *inside* the scale, which is what keeps `score_scale`
        #     a pure temperature. The decision boundary is `scores = -bias`
        #     either way round, but only this way is it independent of the
        #     scale: `s * (u + b)` crosses zero where `u = -b` whatever `s` is,
        #     where `s * u + b` crosses at `u = -b/s` and so moves the boundary
        #     every time the listener's confidence changes -- which would make
        #     `b` something the head has to re-learn against `s` rather than
        #     once. `train.py` reads the binary decision as `scores > 0` and the
        #     reference-game decision as an argmax, and this leaves both
        #     invariant, as on `BilinearGRUComparer`.
        return self.score_scale * (scores + self.decision.bias)

    def reset_parameters(self):
        # Every submodule holding a parameter, including the two adapters and
        #     the norms. The adapters were missing here once, so a reset
        #     listener kept the projections that map referents and messages
        #     into `d_model` -- most of what it had learned about its inputs --
        #     while everything downstream of them was re-drawn. The
        #     parameter-free norms are listed for the mirror reason: turning
        #     `elementwise_affine` back on must not leave a reset listener
        #     holding trained gains.
        self.referent_adapter.reset_parameters()
        self.referent_layer_norm.reset_parameters()
        self.message_adapter.reset_parameters()
        self.message_cross_attention.reset_parameters()
        self.message_residual_norm.reset_parameters()
        self.encoding.reset_parameters()
        self.referent_cross_attention.reset_parameters()
        self.referent_residual_norm.reset_parameters()
        self.referent_self_attention.reset_parameters()
        self.referent_self_attention_norm.reset_parameters()
        self.decision_layer_norm.reset_parameters()
        self.decision.reset_parameters()
        nn.init.zeros_(self.log_score_scale)


class Receiver(nn.Module):
    def __init__(self, feature_model, token_embedding_module, comparer):
        """
        Note there is no `vision_dropout` here, unlike `Sender`. The listener's
            regularisation lives entirely in the comparer, which masks the
            referent and message embeddings equally off a single `dropout`.
            A dropout here would land on the same tensor the comparer masks,
            with nothing but a reshape between the two, so the pair would
            silently compose into one mask at a rate neither knob names.
            `Sender.vision_dropout` is not redundant in the same way: the
            prototyper pools between it and `prototype_dropout`, which is
            exactly what makes the pre-pool mask the weaker of the two.
        """
        super().__init__()
        self.feature_model = feature_model
        self.token_embedding = token_embedding_module
        self.comparer = comparer

    def forward(self, referents, messages):
        batch_size = referents.shape[0]
        n_obj = referents.shape[1]
        rest = referents.shape[2:]

        # Embed the referents
        referents_flat = referents.view(batch_size * n_obj, *rest)
        embedded_referents = self.feature_model(referents_flat)
        embedded_referents = embedded_referents.view(batch_size, n_obj, -1)

        # Embed the messages
        messages = messages @ self.token_embedding.weight

        return self.comparer(embedded_referents, messages)

    def reset_parameters(self):
        self.feature_model.reset_parameters()
        self.token_embedding.reset_parameters()
        self.comparer.reset_parameters()