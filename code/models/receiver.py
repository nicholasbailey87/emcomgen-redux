"""
Listener models
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
        Compare an embedded message against a set of candidate referents with
            two decoder stacks, each reading the other's stream as memory.

        `message_decoder` runs `message_layers` blocks of self-attention,
            cross-attention into the candidate set, and a feedforward, so the
            meaning it refines is discriminative rather than absolute.
            `referent_decoder` then runs `referent_layers` blocks the other way
            round -- cross-attention into the refined message *first*, then the
            candidates read each other, then a feedforward -- so the message
            crosses into the scored stream once per block rather than once in
            total, and the candidates compare message-informed representations.

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
        self.message_layers = kwargs["message_layers"]
        self.referent_layers = kwargs["referent_layers"]
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

        # One pair per stack, at that stack's own depth. `decoder=True` counts
        #     three sublayers to the block, which is what these are, so what is
        #     passed is the block count and not a multiple of it. See
        #     docs/broccoli.md.
        self.message_alpha, self.message_beta = (
            model_util.resolve_residual_scaling(
                kwargs["alpha"], kwargs["beta"], self.message_layers,
                decoder=True,
            )
        )
        self.referent_alpha, self.referent_beta = (
            model_util.resolve_residual_scaling(
                kwargs["alpha"], kwargs["beta"], self.referent_layers,
                decoder=True,
            )
        )

        # Suppressed unless a stack is deep enough for a depth ramp to mean
        #     anything, and asked of each stack separately for the same reason
        #     the scalings are.
        self.message_stochastic_depth = (
            kwargs["stochastic_depth"] if self.message_layers > 1 else 0.0
        )
        self.referent_stochastic_depth = (
            kwargs["stochastic_depth"] if self.referent_layers > 1 else 0.0
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

        # The message reads the candidate set. `causal` follows
        #     `bidirectional` exactly as the encoder it replaces did: the
        #     message arrives whole, so nothing here masks it left to right.
        #     The referents are this stack's memory, which is what
        #     `referent_layer_norm` above is for -- a post-norm stack
        #     normalises its own stream and never its memory.
        self.message_decoder = transformer_decoder.TransformerDecoder(
            self.message_length,
            # `memory_len` is recorded for the caller and sizes nothing; the
            #     candidate count is a property of the game, not of this module.
            None,
            self.d_model,
            self.message_layers,
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
            stochastic_depth=self.message_stochastic_depth,
            depthwise_linear_stochastic_depth=self.depthwise_linear_stochastic_depth,
            linear_module=nn.Linear,
            bos_tokens=self.utility_tokens,
            knocking_heads=self.knocking_heads,
            return_bos_tokens=self.return_bos_tokens,
            pre_norm=self.pre_norm,
            post_norm=self.post_norm,
            msa_scaling="d",
            alpha=self.message_alpha,
            beta=self.message_beta,
            causal=not self.bidirectional,
            cross_first=False,
        )

        # Each candidate reads the refined message, then the candidates read
        #     each other -- `cross_first`, so what the self-attention compares
        #     has already been informed by the message. That self-attention is
        #     the only stage at which a score can depend on the rest of the set,
        #     and the only route to the concept game's clustering shortcut.
        #
        # `causal=False` is not negotiable, and `relative_position_embedding`
        #     is False for the same reason: referent order is the label vector,
        #     so anything able to index its own sequence axis could ignore the
        #     message. `test_no_stage_can_read_the_referent_ordering` pins it.
        self.referent_decoder = transformer_decoder.TransformerDecoder(
            # `seq_len` sizes the causal mask, which is off here, and the
            #     absolute position embedding, which is not built.
            None,
            self.message_length,
            self.d_model,
            self.referent_layers,
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
            stochastic_depth=self.referent_stochastic_depth,
            depthwise_linear_stochastic_depth=self.depthwise_linear_stochastic_depth,
            linear_module=nn.Linear,
            bos_tokens=0,
            knocking_heads=False,
            return_bos_tokens=False,
            pre_norm=self.pre_norm,
            post_norm=self.post_norm,
            msa_scaling="d",
            alpha=self.referent_alpha,
            beta=self.referent_beta,
            causal=False,
            cross_first=True,
        )

        # A bias, and a weight whose magnitude is free. Both were removed once
        #     and restored; see docs/anecdotes.md. It reads straight off the
        #     referent stack's last post-norm, which is an `RMSNorm` and so
        #     already equalises the candidates' lengths -- no object can be read
        #     loudly for being large. That norm's gain is learnable, so global
        #     volume stays free, which is the route `decision_spread` watches.
        self.decision = nn.Linear(self.d_model, 1)

        # Metrics only: set on every `forward`, read by `train.py`. The standard
        #     deviation of the scores. A monotone descent towards zero is the
        #     finding, not wandering. See docs/measurement.md.
        self.decision_spread = float("nan")

        # Excess kurtosis of the scores, and the column to read first. Negative
        #     means bimodal, which is what discriminating looks like; sustained
        #     positive alongside chance accuracy is a listener with nothing to
        #     say. See docs/measurement.md.
        self.decision_kurtosis = float("nan")

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

        # 1. The message reads the candidate set, block by block, so what
        #    comes out is a discriminative meaning rather than an absolute one.
        encoded_messages = self.message_decoder(messages, referents)

        # 2. Each candidate reads the refined message, then the candidates read
        #    each other, once per block. This stack's memory arrives normalised
        #    from the stack above's own last post-norm.
        refined = self.referent_decoder(referents, encoded_messages)

        # 3. Read out. That last post-norm is an `RMSNorm`, so the candidates
        #    reach this at equal length and no object can be read loudly for
        #    being large.
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
        self.message_decoder.reset_parameters()
        self.referent_decoder.reset_parameters()
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