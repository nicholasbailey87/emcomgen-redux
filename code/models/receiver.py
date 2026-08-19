"""
Listener models
"""

import math

import torch
import torch.nn as nn

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
        # `dropout` means exactly one thing across both comparers: mask *both*
        #     operands of the comparison, i.e. the pooled message embedding and
        #     the incoming referent embeddings. It is the listener's only
        #     regulariser and the counterpart of the sender's
        #     `prototype_dropout`; like it, it sits where there is no averaging
        #     left downstream to restore the masked units. Module internals are
        #     fixed constants below, so raising this knob never silently
        #     rewires the architecture.
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
        projected = self.dropout(projected)

        # Both operands of the comparison are regularised, not just the
        #     message. The score is a dot product, so dropping units of one
        #     side only lets the listener lean on whichever side is left
        #     intact; masking both forces the bilinear map to be robust in the
        #     referent basis as well as the message basis.
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
        Use multiheaded cross-attention as per Attention Is All You Need
            (https://arxiv.org/abs/1706.03762) to compare embedded messages
            with sets of possible message referents.
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
        # This comparer's residual path is two stacks with a cross-attention
        #     between them, so the depth `alpha` and `beta` should be derived
        #     from is each sub-stack's own, not the configured total. Resolved
        #     separately for that reason; a pinned number still passes straight
        #     through to both. `layers` is split unevenly when it is odd, the
        #     reading stack taking the extra block.
        self.encoding_layers = self.layers - (self.layers // 2)
        self.fusion_layers = int(self.layers // 2)

        self.encoding_alpha, self.encoding_beta = (
            model_util.resolve_residual_scaling(
                kwargs["alpha"], kwargs["beta"], self.encoding_layers
            )
        )

        # A fusion stack of no blocks has no residual path for these to scale,
        #     so they are inert and the derivation has nothing to derive from.
        self.fusion_alpha, self.fusion_beta = (
            model_util.resolve_residual_scaling(
                kwargs["alpha"], kwargs["beta"], self.fusion_layers
            )
            if self.fusion_layers >= 1
            else (1.0, 1.0)
        )
        # Suppressed unless the fusion stack is deep enough for a depth ramp to
        #     mean anything: `depthwise_linear_stochastic_depth` spreads the
        #     rate linearly across layers, so a one-layer stack would get a
        #     single rate of 0.0 regardless and a two-layer stack only half the
        #     configured rate on its second block.
        self.stochastic_depth = (
            kwargs["stochastic_depth"] if int(self.layers // 2) > 1 else 0.0
        )

        # Same meaning as in `BilinearGRUComparer`: mask both operands of the
        #     comparison. Applied to each input as it arrives, before the
        #     adapters, which is the only point at which the two are
        #     symmetrically placed. Attention dropout is a separate setting
        #     (`receiver_comparer.cross_attention_dropout`) rather than this knob.
        self.input_dropout = nn.Dropout(p=self.dropout)

        self.referent_adapter = nn.Linear(
            self.referent_embedding_size,
            self.d_model
        )

        # Parameter-free, as on `BilinearGRUComparer`: an affine here would
        #     be a second route to score magnitude alongside `decision`, and
        #     the point of the norm is that the referent arrives at a size
        #     the vision model did not choose.
        self.referent_layer_norm = nn.LayerNorm(
            self.d_model, elementwise_affine=False, eps=LAYER_NORM_EPS
        )

        self.message_adapter = nn.Linear(
            self.token_embedding_size,
            self.d_model
        )

        self.encoding = broccoli.transformer.TransformerEncoder(
            self.message_length, # seq_len can be none as length-invariant
            self.d_model,
            self.encoding_layers,
            self.heads,
            absolute_position_embedding=self.absolute_position_embedding,
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

        self.cross_attention = broccoli.transformer.MHAttention(
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
            causal=False,
            seq_len=self.message_length,
            linear_module=nn.Linear,
            bos_tokens=0,
            knocking_heads=False,
            # No positional information here: the message side already carries
            #     it from `self.encoding`, and the referent side is an
            #     unordered set. `positional_heads` is inert while
            #     `rotary_embedding` is None, but is pinned anyway — note that
            #     broccoli defaults it to 0.25 on `MHAttention` and 0.5 on
            #     `TransformerEncoder`, so the two are not interchangeable.
            rotary_embedding=None,
            positional_heads=0.25,
            source_size=None,
            scaling="d",
        )

        # Fusion module to refine the cross-attention output
        # This makes use of the position embeddings from the two encoders,
        #     so doesn't need its own position embeddings and is seq_len-invariant
        self.fusion = broccoli.transformer.TransformerEncoder(
            None, # seq_len can be none as length-invariant
            self.d_model,
            self.fusion_layers,
            self.heads,
            # Not configurable, unlike the same arguments on `self.encoding`
            #     above, and pinned rather than exposed because turning either
            #     on would break the module.
            #
            # Note this block's sequence axis is the *referent set*, not the
            #     message. `self.encoding` is the one that runs over the
            #     message, and that is where message position is embedded; by
            #     the time the cross-attention has run, message position has
            #     been summarised into each referent's vector and the axis
            #     here is referents. So a position embedding here would number
            #     the referents.
            #
            # The weak reason not to is that a set has no order to embed. The
            #     load-bearing reason is that in this codebase the referent
            #     order *is* the label vector: `data.util.split_spk_lis`
            #     writes positives into the first half of each agent's view and
            #     negatives into the second, and the augmentation in
            #     `ConceptDataset.__getitem__` (and `CUBDataset.sample_game`)
            #     permutes only *within* each half. `Sender.get_prototypes`
            #     relies on the same arrangement and raises without it.
            #
            # A fusion stack that could index its own sequence axis could
            #     therefore learn "the first half are targets" and score
            #     perfectly while ignoring the message. Without position
            #     embeddings, and with `causal=False` below, this block is
            #     permutation-equivariant over referents and cannot read the
            #     ordering at all. `BilinearGRUComparer` is immune for a
            #     different reason: it scores each referent in isolation and
            #     never sees the set.
            #
            # Turning either on would also need a `source_size` for the
            #     referent axis, which is a property of the game rather than of
            #     this module.
            absolute_position_embedding=False,
            relative_position_embedding=False,
            # Inert while both position embeddings are False, just below; pinned
            #     to the repo-wide 1.0 rather than left to broccoli's 0.5 so that
            #     turning either of them on could not quietly reintroduce a head
            #     partition. See `ViT2` for why 1.0.
            positional_heads=1.0,
            source_size=None,
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
            causal=False,
            linear_module=nn.Linear,
            bos_tokens=self.utility_tokens,
            knocking_heads=self.knocking_heads,
            return_bos_tokens=self.return_bos_tokens,
            pre_norm=self.pre_norm,
            post_norm=self.post_norm,
            msa_scaling="d",
            alpha=self.fusion_alpha,
            beta=self.fusion_beta,
        )

        self.decision = nn.Linear(self.d_model, 1, bias=True)

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
        # Both operands masked, symmetrically, before their adapters.
        referents = self.input_dropout(referents)
        messages = self.input_dropout(messages)

        referents = self.referent_adapter(referents)
        normed_referents = self.referent_layer_norm(referents)
        messages = self.message_adapter(messages)
        encoded_messages = self.encoding(messages)
        mixed = self.cross_attention(
            normed_referents,
            encoded_messages,
            encoded_messages
        )
        refined = self.fusion(mixed)
        scores = self.decision(refined) # (batch, n_objects, 1)
        return scores.squeeze(-1) # (batch, n_objects)

    def reset_parameters(self):
        # The two adapters and the referent norm were missing here, so a reset
        #     listener kept the projections that map referents and messages into
        #     `d_model` -- i.e. most of what it had learned about its inputs --
        #     while everything downstream of them was re-drawn.
        self.referent_adapter.reset_parameters()
        self.referent_layer_norm.reset_parameters()
        self.message_adapter.reset_parameters()
        self.encoding.reset_parameters()
        self.fusion.reset_parameters()
        self.cross_attention.reset_parameters()
        self.decision.reset_parameters()


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