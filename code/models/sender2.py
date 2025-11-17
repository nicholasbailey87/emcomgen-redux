"""
Speaker models
"""

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.distributions import Gumbel
from torch.nn.utils.rnn import pad_sequence

from einops import einsum
from einops.layers.torch import Rearrange

import data
import data.language

import broccoli


class AveragePrototyper(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, samples, labels=None):
        n_pos_ex = samples.size(1) // 2

        positive_examples = samples[:, :n_pos_ex, :]
        negative_examples = samples[:, n_pos_ex:, :]

        positive_prototype = positive_examples.mean(1)
        negative_prototype = negative_examples.mean(1)

        return positive_prototype, negative_prototype

    def reset_parameters(self):
        pass


class AttentionPrototyper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_pool = broccoli.vit.SequencePool(config['sender']['arguments']['d_model'])
        self.neg_pool = broccoli.vit.SequencePool(config['sender']['arguments']['d_model'])

    def forward(self, samples, labels=None):
        n_pos_ex = samples.size(1) // 2

        positive_examples = samples[:, :n_pos_ex, :]
        negative_examples = samples[:, n_pos_ex:, :]

        positive_prototype = self.pos_pool(positive_examples)
        negative_prototype = self.neg_pool(negative_examples)

        return positive_prototype, negative_prototype

    def reset_parameters(self):
        super().reset_parameters()
        self.pos_pool.reset_parameters()
        self.neg_pool.reset_parameters()

class SenderGRULM(nn.Module):
    def __init__(
          self,
          embedding_dim,
          hidden_size,
          vocab_size,
          feat_size
        ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.outputs2vocab = nn.Linear(hidden_size, vocab_size)
        self.init_h = nn.Linear(2 * feat_size, hidden_size)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(
        self,
        prototypes,
        **kwargs
    ):
        # Initialize hidden state (this shape is always (num_layers, B, H))
        concatenate_prototypes = torch.cat(prototypes, 1)
        states = self.init_h(concatenate_prototypes).unsqueeze(0) # (1, B, H)

        batch_size = states.size(1)
        lang = []

        # Create and add SOS token
        sos_onehot = torch.zeros(batch_size, 1, self.vocab_size).to(states.device)
        sos_onehot[:, 0, data.language.SOS_IDX] = 1.0
        lang.append(sos_onehot)

        inputs = sos_onehot # Shape: (B, 1, V)
        inputs = inputs @ self.embedding.weight  # Shape: (B, 1, D)

        # Main sampling loop (fixed length of max_len - 2)
        for i in range(self.max_len - 2):

            # Input is (B, 1, D), Output is (B, 1, H)
            outputs, states = self.gru(inputs, states)

            outputs = outputs.squeeze(1) # Shape: (B, H)
            outputs = self.outputs2vocab(outputs) # Shape: (B, V)

            if self.greedy:
                predicted_indices = outputs.max(1)[1]
                predicted_onehot = F.one_hot(predicted_indices, self.vocab_size).float()
            else:
                outputs = F.gumbel_softmax(outputs, tau=self.softmax_temp, hard=False)

                if self.uniform_weight != 0.0:
                    uniform_outputs = torch.full_like(
                        outputs,
                        1 / outputs.shape[-1]
                    )
                    outputs = (
                        (1 - uniform_weight) * outputs
                        +
                        uniform_weight * uniform_outputs
                    )
                
                predicted_onehot = (
                    torch.zeros_like(outputs)
                        .scatter_(
                            -1,
                            torch.argmax(outputs, dim=-1).unsqueeze(-1),
                            1.
                        )
                )

                predicted_onehot = predicted_onehot - outputs.detach() + outputs

                # if np.random.random() < self.eps:
                #     random_i = torch.randint(outputs.shape[1], (outputs.shape[0], 1)).to(predicted_onehot.device)
                #     random_onehot = torch.zeros_like(predicted_onehot).scatter_(-1, random_i, 1.0)
                #     predicted_onehot = (random_onehot - predicted_onehot.detach()) + predicted_onehot

            lang.append(predicted_onehot.unsqueeze(1)) # (B, 1, V)

            inputs = (predicted_onehot.unsqueeze(1)) @ self.embedding.weight # (B, 1, D)

        # Add final EOS token
        eos_onehot = torch.zeros(batch_size, 1, self.vocab_size).to(states.device)
        eos_onehot[:, 0, data.language.EOS_IDX] = 1.0
        lang.append(eos_onehot)

        # Concatenate along the sequence dim (1)
        lang_tensor = torch.cat(lang, 1) # (B, max_len, V)

        return lang_tensor

    def reset_parameters(self):
        super().reset_parameters()
        self.init_h.reset_parameters()
        self.gru.reset_parameters()
        self.outputs2vocab.reset_parameters()
        self.embedding.reset_parameters()


class SenderTransformerLM(nn.Module):
    def __init__(
          self,
          d_model,
          dropout,
          max_len=4,
          vocab_size=8,
          layers=7,
          heads=4,
          softmax_temp=1.0,
          causal=False
        ):
        super().__init__()

        if max_len < 2:
            raise ValueError("max_len must be at least 2 (for SOS and EOS tokens)")

        self.max_len = max_len
        self.vocab_size = vocab_size
        self.heads = heads
        self.causal = causal
        self.softmax_temp = softmax_temp

        self.content_length = self.max_len - 2

        self.query = nn.Parameter(torch.empty(self.content_length, d_model))
        nn.init.normal_(self.query, mean=0.0, std=1.0)

        self.cross_attention = broccoli.transformer.MHAttention(
            d_model,
            heads,
            dropout=dropout,
            causal=causal,
            seq_len=self.content_length,
            scaling="d",
        )

        self.transformer = broccoli.transformer.TransformerEncoder(
            self.content_length,
            d_model,
            layers,
            heads,
            absolute_position_embedding=True,
            relative_position_embedding=True,
            source_size=(self.content_length,),
            mlp_ratio=2,
            activation=broccoli.activation.SwiGLU,
            activation_kwargs=None,
            mlp_dropout=0.,
            msa_dropout=0.,
            stochastic_depth=0.2,
            causal=self.causal,
            bos_tokens=self.heads ** 2,
            return_bos_tokens=False,
        )
        self.outputs2vocab = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        prototypes,
        **kwargs
    ):
        batch_size = prototypes[0].size(0)
        device = prototypes[0].device

        stack_prototypes = torch.stack(prototypes, 1)

        query = self.query.unsqueeze(0).expand(
            batch_size,
            -1,
            -1
        )

        input = self.cross_attention(query, stack_prototypes, stack_prototypes)
        outputs = self.transformer(input)
        logits = self.outputs2vocab(outputs)
        outputs = F.gumbel_softmax(logits, tau=self.softmax_temp, hard=False)

        if uniform_weight != 0.0:
            uniform_outputs = torch.full_like(
                outputs,
                1 / outputs.shape[-1]
            )
            outputs = (
                (1 - uniform_weight) * outputs
                +
                uniform_weight * uniform_outputs
            )
                
        onehot_content = (
            torch.zeros_like(outputs)
                .scatter_(
                    -1,
                    torch.argmax(outputs, dim=-1).unsqueeze(-1),
                    1.
                )
        )

        onehot_content = onehot_content - outputs.detach() + outputs

        sos_onehot = torch.zeros(batch_size, 1, self.vocab_size, device=device)
        sos_onehot[:, 0, data.language.SOS_IDX] = 1.0
        eos_onehot = torch.zeros(batch_size, 1, self.vocab_size, device=device)
        eos_onehot[:, 0, data.language.EOS_IDX] = 1.0

        onehot = torch.cat([sos_onehot, onehot_content, eos_onehot], dim=1)

        return onehot # (batch, max_len, vocabulary)

    def reset_parameters(self):
        nn.init.normal_(self.query, mean=0.0, std=1.0)
        self.cross_attention.reset_parameters()
        self.transformer.reset_parameters()
        self.outputs2vocab.reset_parameters()


class Sender(nn.Module):
    def __init__(
        self,
        feat_model: nn.Module,
        prototyper: nn.Module,
        language_model: nn.Module,
        dropout: float= 0.5
    ):
        """
        An agent that will receive one or more positive examples of a concept and
            one or more negative examples of a concept and will produce an utterance
            intended to communicate the concept

        Args:
            feat_model: The module used to produce embeddings from referents
            prototyper: The module used to create prototypes from positive and
                negative examples of referents
            language_model: The module used to create utterances based on prototypes
            dropout: Dropout probability between the `feat_model` and `prototyper`
        """
        super().__init__()
        self.feat_model = feat_model
        self.feat_size = feat_model.final_feat_dim
        self.prototyper = prototyper
        self.language_model = language_model
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        samples,
        targets,
        **kwargs
    ):
        """Sample from image features"""
        # Flatten and embed samples
        batch_size = samples.shape[0]
        n_obj = samples.shape[1]
        rest = samples.shape[2:]
        flat_samples = samples.view(batch_size * n_obj, *rest)
        embedded_samples = self.dropout(self.feat_model(flat_samples))
        embedded_samples = embedded_samples.view(batch_size, n_obj, -1)

        prototypes = self.prototyper(embedded_samples, targets)

        prototypes_concat = torch.cat(prototypes, 1)

        messages = self.language_model(
            prototypes,
            **kwargs
        )

        return messages, prototypes_concat

    def reset_parameters(self):
        self.feat_model.reset_parameters()
        self.prototyper.reset_parameters()
        self.language_model.reset_parameters()