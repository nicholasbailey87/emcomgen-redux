"""
Listener models
"""

import torch
import torch.nn as nn

import broccoli

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
        self.dropout = nn.Dropout(p=kwargs["dropout"])
        self.bidirectional = kwargs["bidirectional"]
        self.layers = kwargs["layers"]

        self.gru = nn.GRU(
            self.token_embedding_size,
            self.d_model,
            num_layers=self.layers,
            bias=True,
            batch_first=True,
            dropout=kwargs["dropout"],
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

        # XXX: Getting the sequence embedding like this could be suboptimal for unidirectional GRU if sequences are padded at the end
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
            mlp_ratio=2,
            activation = broccoli.activation.SwiGLU,
            stochastic_depth=self.stochastic_depth,
            causal=not self.bidirectional,
            utility_tokens=self.utility_tokens,
            return_utility_tokens=False,
            pre_norm=True,
            post_norm=True,
            normformer=True,
            msa_scaling="d",
            checkpoint_ff=True,
        )

        self.cross_attention = broccoli.transformer.MHAttention(
            self.d_model,
            self.heads,
            dropout=self.dropout,
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
            mlp_ratio=2,
            activation = broccoli.activation.SwiGLU,
            stochastic_depth=self.stochastic_depth,
            causal=False,
            utility_tokens=self.utility_tokens,
            return_utility_tokens=False,
            pre_norm=True,
            post_norm=True,
            normformer=True,
            msa_scaling="d",
            checkpoint_ff=True,
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
    def __init__(self, feature_model, token_embedding_module, comparer, vision_dropout):
        super().__init__()
        self.feature_model = feature_model
        self.token_embedding = token_embedding_module
        self.comparer = comparer
        self.vision_dropout = nn.Dropout(p=vision_dropout)

    def forward(self, referents, messages):
        batch_size = referents.shape[0]
        n_obj = referents.shape[1]
        rest = referents.shape[2:]

        # Embed the referents
        referents_flat = referents.view(batch_size * n_obj, *rest)
        embedded_referents = self.feature_model(referents_flat)
        embedded_referents = self.vision_dropout(embedded_referents)
        embedded_referents = embedded_referents.view(batch_size, n_obj, -1)

        # Embed the messages
        messages = messages @ self.token_embedding.weight

        return self.comparer(embedded_referents, messages)

    def reset_parameters(self):
        self.feature_model.reset_parameters()
        self.token_embedding.reset_parameters()
        self.comparer.reset_parameters()