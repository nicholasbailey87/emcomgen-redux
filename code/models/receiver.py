"""
Listener models
"""

import torch
import torch.nn as nn

import broccoli

from . import model_util

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
        
        message_embeddings = self.dropout(message_embeddings)
        projected = self.bilinear(message_embeddings)

        # Both operands of the comparison are regularised, not just the
        #     message. The score is a dot product, so dropping units of one
        #     side only lets the listener lean on whichever side is left
        #     intact; masking both forces the bilinear map to be robust in the
        #     referent basis as well as the message basis.
        referents = self.dropout(referents)

        scores = torch.einsum("ijh,ih->ij", (referents, projected)) # (batch, n_objects)

        return scores

    def reset_parameters(self):
        self.gru.reset_parameters()
        self.bilinear.reset_parameters()


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
        self.ff_ratio = kwargs["ff_ratio"]
        self.cross_attention_dropout = kwargs["cross_attention_dropout"]
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
        self.alpha = kwargs["alpha"]
        self.beta = kwargs["beta"]
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

        self.referent_layer_norm = nn.LayerNorm(self.d_model)

        self.message_adapter = nn.Linear(
            self.token_embedding_size,
            self.d_model
        )

        self.encoding = broccoli.transformer.TransformerEncoder(
            self.message_length, # seq_len can be none as length-invariant
            self.d_model,
            self.layers - (self.layers // 2),
            self.heads,
            absolute_position_embedding=self.absolute_position_embedding,
            relative_position_embedding=self.relative_position_embedding,
            positional_heads=self.positional_heads,
            # Derived from the data, not configured separately: this block
            #     reads the message, so its source is the message length.
            source_size=(self.message_length,),
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
            int(self.layers // 2),
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
            positional_heads=self.positional_heads, # inert while both are False
            source_size=None,
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
            causal=False,
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