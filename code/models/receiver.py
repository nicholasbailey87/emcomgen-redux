"""
Listener models
"""

import math

import torch
import torch.nn as nn

import broccoli

from . import model_util

# Mirrors `sender.LAYER_NORM_EPS`, and load-bearing for the same reason: below
#     the 1e-5 default the normaliser quietly stops normalising and the score's
#     magnitude goes back to the backbone. See docs/channel.md.
LAYER_NORM_EPS = 1e-12

# Every broccoli module below is constructed with its full argument list, even
#     where an argument is inert under the current settings, because broccoli's
#     defaults are not a stable interface. See docs/broccoli.md.

class BilinearGRUComparer(nn.Module):
    def __init__(
        self,
        referent_embedding_size,
        **kwargs
    ):
        """
        Compare embedded messages with sets of possible referents by a bilinear
            score: `score = obj_emb.T @ weights @ m_emb`.

        The projection has no bias, so the score depends only on the
            relationship between message and object; a bias would add a
            message-independent per-object prior. See docs/architecture.md.
        """
        super().__init__()
        self.referent_embedding_size = referent_embedding_size
        self.token_embedding_size = kwargs["token_embedding_size"]
        self.d_model = kwargs["d_model"]
        # Masks the incoming referent embeddings only, after their norm and
        #     immediately before the comparison. The message operand is left
        #     alone -- it already arrives through the Gumbel channel. See
        #     docs/architecture.md.
        self.dropout = nn.Dropout(p=kwargs["dropout"])
        self.bidirectional = kwargs["bidirectional"]
        self.layers = kwargs["layers"]

        self.gru = nn.GRU(
            self.token_embedding_size,
            self.d_model,
            num_layers=self.layers,
            bias=True,
            batch_first=True,
            # Fixed at 0.0 to match jayelm, and inert while `layers = 1`. See
            #     docs/broccoli.md.
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
        #     at 1.0, stored as a log. See docs/architecture.md.
        self.log_score_scale = nn.Parameter(torch.zeros(()))

    @property
    def score_scale(self):
        """
        The multiplier applied to the normalised message operand, always
            positive. Read here rather than exponentiating at the use site, so
            `forward` and the metrics column cannot drift apart.
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
        
        # Normalised *after* the projection: `bilinear` is free, so a norm on
        #     its input would set only where `W` starts.
        projected = self.bilinear(message_embeddings)
        projected = self.message_layer_norm(projected) * self.score_scale

        # The referents are masked and the message is not; see
        #     docs/architecture.md.
        referents = self.referent_layer_norm(referents)
        referents = self.dropout(referents)

        scores = torch.einsum("ijh,ih->ij", (referents, projected)) # (batch, n_objects)

        # Attention's `1/sqrt(d)`, and load-bearing now that both operands are
        #     normalised: it is what makes `score_scale = 1.0` the calibrated
        #     opening at any `referent_embedding_size`.
        return scores / math.sqrt(self.referent_embedding_size)

    def reset_parameters(self):
        self.gru.reset_parameters()
        self.bilinear.reset_parameters()
        # No-ops while the two norms are parameter-free, and listed anyway so
        #     that turning `elementwise_affine` back on cannot leave a reset
        #     listener holding trained gains.
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

        `decision` is a plain `nn.Linear(d_model, 1)`, so the listener can turn
            its own volume down and BCE rewards it for doing so. That route is
            deliberately open, and watched by `decision_spread` rather than
            closed -- two attempts to close it are recorded in
            docs/anecdotes.md, along with why the second was reverted.

        See docs/architecture.md for the rest.
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

        # `layers` is the message encoder's depth, and nothing else's.
        self.encoding_alpha, self.encoding_beta = (
            model_util.resolve_residual_scaling(
                kwargs["alpha"], kwargs["beta"], self.layers
            )
        )

        # One pair for the three hand-written residuals, resolved at depth 1:
        #     bare attention sublayers are one layer's worth of residual path.
        #     See docs/broccoli.md.
        self.residual_alpha, self.residual_beta = (
            model_util.resolve_residual_scaling(
                kwargs["alpha"], kwargs["beta"], 1
            )
        )

        # Suppressed unless the encoder is deep enough for a depth ramp to mean
        #     anything. The three residuals below get none -- they are not
        #     `EncoderBlock`s and have no branch to drop.
        self.stochastic_depth = (
            kwargs["stochastic_depth"] if self.layers > 1 else 0.0
        )

        # The listener's regulariser, masking the referents only, and placed
        #     after the norm rather than before the adapter. Attention dropout
        #     is a separate setting. See docs/architecture.md.
        self.input_dropout = nn.Dropout(p=self.dropout)

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
        #     for being large rather than for matching. See docs/architecture.md.
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
            # Pinned False, and no longer a config option; every stack here
            #     runs rotary. See docs/broccoli.md.
            absolute_position_embedding=False,
            relative_position_embedding=self.relative_position_embedding,
            # Pinned at 1.0, not configurable -- see `ViT2` for the argument.
            positional_heads=1.0,
            # Derived from the data, not configured separately: this block
            #     reads the message, so its source is the message length.
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
        #     which a score can depend on the rest of the set.
        self.referent_self_attention = self._attention()
        self.referent_self_attention_norm = nn.RMSNorm(self.d_model)

        # Parameter-free, and not optional: it is the only thing between the
        #     preceding post-norm's learnable gain and global score volume, and
        #     it equalises the candidates' lengths so `scores` is an angle. See
        #     docs/architecture.md.
        self.decision_layer_norm = nn.LayerNorm(
            self.d_model, elementwise_affine=False, eps=LAYER_NORM_EPS
        )

        # A bias, and a weight whose magnitude is free. Both were removed once
        #     and restored; `decision_layer_norm` above is the only structural
        #     guard left, which is why its affine-free flag is asserted in
        #     `test_the_cross_attention_norms_that_must_be_affine_free_are`
        #     rather than merely commented. See docs/anecdotes.md.
        self.decision = nn.Linear(self.d_model, 1)

        # Metrics only: set on every `forward`, read by `train.py`. The standard
        #     deviation of the scores, opening around 0.57. A monotone descent
        #     towards zero is the finding, not wandering. See
        #     docs/measurement.md.
        self.decision_spread = float("nan")

        # Excess kurtosis of the scores, and the column to read first. Negative
        #     means bimodal, which is what discriminating looks like; sustained
        #     positive alongside chance accuracy is a listener with nothing to
        #     say. See docs/measurement.md.
        self.decision_kurtosis = float("nan")

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
            # Architecture internals, deliberately not `self.dropout`. See
            #     docs/broccoli.md.
            dropout=self.cross_attention_dropout,
            # Inert while `causal=False`, and that is not negotiable: referent
            #     order is the label vector, so anything able to index its own
            #     sequence axis could ignore the message. See
            #     docs/architecture.md.
            causal=False,
            seq_len=self.message_length,
            linear_module=nn.Linear,
            bos_tokens=0,
            knocking_heads=False,
            # No positional information in any of the three; `positional_heads`
            #     is inert here and pinned at the repo-wide 1.0. See
            #     docs/architecture.md.
            rotary_embedding=None,
            positional_heads=1.0,
            source_size=None,
            scaling="d",
        )

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
        #    `TransformerEncoder.preprocess` adds its position embedding in
        #    place. Harmless because nothing reads `messages` again -- take a
        #    copy first if a second residual is ever added.
        encoded_messages = self.encoding(messages)

        # 3. Each candidate reads the refined message. The residual is what
        #    carries referent identity to the readout linearly.
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

        # 5. Read out. Every candidate is at norm `sqrt(d)` after
        #    `decision_layer_norm`, so `scores` is an angle and nothing else --
        #    no object can be read loudly for being large.
        refined = self.decision_layer_norm(refined)
        scores = self.decision(refined).squeeze(-1) # (batch, n_objects)

        # `.item()` in a forward pass costs a sync and a graph break under
        #     `torch.compile`, which is on. Paid deliberately -- a metric nobody
        #     can read is how the last collapse ran unnoticed.
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

        # Returned as they are: the volume lives in `decision.weight` and is
        #     learned. Note `train.py` reads `lis_scores > 0`, so accuracy is
        #     invariant to any positive rescale of the readout, which is why the
        #     accuracy column cannot see a volume collapse.
        return scores

    def reset_parameters(self):
        # Every submodule holding a parameter, including the two adapters (which
        #     were missing here once) and the parameter-free norms. See
        #     docs/anecdotes.md.
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


class Receiver(nn.Module):
    def __init__(self, feature_model, token_embedding_module, comparer):
        """
        Note there is no `vision_dropout` here, unlike `Sender`: the listener's
            regularisation lives in the comparer, and a dropout here would land
            on the same tensor with nothing but a reshape between the two. See
            docs/architecture.md.
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