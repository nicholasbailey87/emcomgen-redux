"""
Listener models
"""

import torch
import torch.nn as nn
# from . import rnn
import broccoli

class BilinearGRUComparer(nn.Module):
    def __init__(
        self,
        referent_embedding_size,
        config
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
        self.token_embedding_size = token_embedding_size
        self.referent_embedding_size = referent_embedding_size
        self.gru_hidden_size = gru_hidden_size
        self.dropout = nn.Dropout(p=dropout)
        self.bilinear = nn.Linear(gru_hidden_size, self.referent_embedding_size, bias=False)
        self.gru = nn.GRU(
            self.token_embedding_size,
            self.gru_hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
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
        token_embeddings, _ = self.gru(messages) # (b, seq_len, d_gru)
        message_embeddings = token_embeddings[:, -1, ...] # (b, d_gru)
        message_embeddings = self.dropout(message_embeddings)
        projected = self.bilinear(message_embeddings)
        scores = torch.einsum("ijh,ih->ij", (referents, message_embeddings))
        return scores # (batch, n_objects)

    def reset_parameters(self):
        self.gru.reset_parameters()
        self.bilinear.reset_parameters()

class TransformerCrossAttentionComparer(nn.Module):
    def __init__(
        self,
        referent_embedding_size,
        config
    ):
        """
        Use multiheaded cross-attention as per Attention Is All You Need
            (https://arxiv.org/abs/1706.03762) to compare embedded messages
            with sets of possible message referents.
        """
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.cross_attention = broccoli.MHAttention(
            d_model,
            n_heads,
            dropout=0.0,
            causal=False,
            seq_len=seq_len,
            scaling="d",
        )

        self.transformer = broccoli.TransformerEncoder(
            seq_len = seq_len,
            d_model = d_model,
            n_layers,
            n_heads,
            absolute_position_embedding=True,
            relative_position_embedding=True,
            source_size=(seq_len,),
            mlp_ratio=2,
            activation = broccoli.activation.SwiGLU,
            stochastic_depth=0.0,
            causal=False,
            bos_tokens=0,
            return_bos_tokens=False,
            pre_norm=True,
            post_norm=True,
            normformer=True,
            msa_scaling="d",
            checkpoint_ff=True,
        )

        self.scorer = nn.Linear(d_model, 1, bias=False)

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
        mixed = self.cross_attention(referents, messages, messages)
        transformed = self.transformer(mixed)
        scores = self.scorer(transformed)
        return scores.squeeze() # (batch, n_objects)

    def reset_parameters(self):
        self.cross_attention.reset_parameters()
        self.transformer.reset_parameters()
        self.scorer.reset_parameters()


class Receiver(nn.Module):
    def __init__(self, feature_model, token_embedding_module, comparer, dropout):
        super().__init__()
        self.feature_model = feature_model
        self.dropout = nn.Dropout(p=dropout)
        self.token_embedding = token_embedding_module
        self.comparer = comparer

    def forward(self, referents, messages):
        # Embed features
        embedded_referents = self.dropout(self.embed_features(feats))

        # Embed language
        messages = messages @ self.token_embedding.weight

        return self.comparer(embedded_referents, messages)

    def reset_parameters(self):
        self.feat_model.reset_parameters()
        self.token_embedding.reset_parameters()
        self.comparer.reset_parameters()