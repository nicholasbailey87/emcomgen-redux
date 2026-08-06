"""
Listener models
"""

import torch
import torch.nn as nn

import broccoli

# Cross-attention dropout for `TransformerCrossAttentionComparer`. A fixed
#     architecture constant, not a tunable: `receiver_comparer.dropout` is
#     reserved for regularising the comparison inputs in both comparers.
MSA_DROPOUT = 0.1


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
        self.stochastic_depth = 0.1 if int(self.layers // 2) > 1 else 0.0

        # Same meaning as in `BilinearGRUComparer`: mask both operands of the
        #     comparison. Applied to each input as it arrives, before the
        #     adapters, which is the only point at which the two are
        #     symmetrically placed. Attention dropout is a fixed constant
        #     (see `MSA_DROPOUT`) rather than this knob.
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
            absolute_position_embedding=True,
            relative_position_embedding=True,
            source_size=(self.message_length,),
            ff_ratio=2,
            activation = broccoli.activation.SwiGLU,
            stochastic_depth=self.stochastic_depth,
            causal=not self.bidirectional,
            bos_tokens=self.utility_tokens,
            return_bos_tokens=False,
            pre_norm=False,
            post_norm=True,
            msa_scaling="d",
        )

        self.cross_attention = broccoli.transformer.MHAttention(
            self.d_model,
            self.heads,
            # Architecture internals, deliberately not `self.dropout` — that
            #     knob is the listener's regulariser on the comparison inputs,
            #     and should not double as an attention-internals setting.
            #     jayelm has no transformer listener to match, so this is
            #     standard practice, alongside the stochastic depth above.
            #     broccoli gates it on `self.training` (`transformer.py:344`),
            #     so it does not leak into eval the way a bare
            #     `F.scaled_dot_product_attention(dropout_p=...)` would.
            dropout=MSA_DROPOUT,
            causal=False,
            seq_len=self.message_length,
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
            absolute_position_embedding=False,
            relative_position_embedding=False,
            source_size=(self.message_length,),
            ff_ratio=2,
            activation = broccoli.activation.SwiGLU,
            stochastic_depth=self.stochastic_depth,
            causal=False,
            bos_tokens=self.utility_tokens,
            return_bos_tokens=False,
            pre_norm=False,
            post_norm=True,
            msa_scaling="d",
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