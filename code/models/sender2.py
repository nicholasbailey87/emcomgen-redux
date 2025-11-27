"""
Speaker models. This includes speakers with a GRU-based language model as
    originally presented in "Emergent Communication of Generalizations"
    (https://arxiv.org/abs/2106.02668) and speakers with causal or non-causal
    Transformer language models. The intention is to show that
    Transformer-based speakers can be just as successful in tasks and show
    equal or greater compositionality.
"""

import warnings

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
# from torch.distributions import Gumbel
# from torch.nn.utils.rnn import pad_sequence

# from einops import einsum
# from einops.layers.torch import Rearrange

import data
import data.language

import broccoli


class AveragePrototyper(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, samples, labels=None):
        """
        Args:
            samples: a tensor of shape (batch, n_examples, embedding size)
                where each example is an embedded image. n_examples must
                be even. The first n_examples / 2 examples are positive
                examples and the remainder are negative examples.
            labels: the labels for the examples provided. This argument
                exists for backwards compatibility, but is not used for
                anything as the first half of provided examples is always
                positive and the second half negative. See `samples` definition.
        """
        n_pos_ex = samples.size(1) // 2

        positive_examples = samples[:, :n_pos_ex, :]
        negative_examples = samples[:, n_pos_ex:, :]

        positive_prototype = positive_examples.mean(1)
        negative_prototype = negative_examples.mean(1)

        return positive_prototype, negative_prototype

    def reset_parameters(self):
        pass


class AttentionPrototyper(nn.Module):
    def __init__(self, d_model, *args, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.pos_pool = broccoli.vit.SequencePool(d_model)
        self.neg_pool = broccoli.vit.SequencePool(d_model)

    def forward(self, samples, labels=None):
        n_pos_ex = samples.size(1) // 2

        positive_examples = samples[:, :n_pos_ex, :]
        negative_examples = samples[:, n_pos_ex:, :]

        positive_prototype = self.pos_pool(positive_examples)
        negative_prototype = self.neg_pool(negative_examples)

        return positive_prototype, negative_prototype

    def reset_parameters(self):
        self.pos_pool.reset_parameters()
        self.neg_pool.reset_parameters()

class SenderGRULM(nn.Module):
    def __init__(
        self,
        referent_embedding_size,
        token_embedding_size,
        d_model,
        vocabulary,
        max_len,
        softmax_temp,
        uniform_weight,
        dropout,
        layers,
        bidirectional
    ):
        super().__init__()
        self.referent_embedding_size = referent_embedding_size
        self.token_embedding_size = token_embedding_size
        self.d_model = d_model
        self.vocabulary = vocabulary
        self.max_len = max_len
        self.softmax_temp = softmax_temp
        self.uniform_weight = uniform_weight
        self.dropout = dropout
        self.layers = layers
        self.bidirectional = bidirectional
        self.directions = 2 if self.bidirectional else 1
        
        self.gru = nn.GRU(
            token_embedding_size,
            d_model,
            num_layers=layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.outputs2vocab = nn.Linear(self.d_model * self.directions, vocabulary)
        self.init_h = nn.Linear(
            2 * referent_embedding_size, 
            self.layers * self.directions * d_model
        )
        
        self.token_embedding = nn.Embedding(vocabulary, token_embedding_size)

    def forward(
        self,
        prototypes,
        **kwargs
    ):
        """
        We don't include options for greedy or epsilon-greedy generation as
            the former is only used in the parts of the code that relate to
            ACRe and the latter are by default not used (and are not
            commented upon in the original paper).
        """
        batch_size = prototypes[0].size(0)

        # Initialize hidden state. Must be (num_layers * directions, B, H)
        concatenate_prototypes = torch.cat(prototypes, 1)
        
        states = (
            self.init_h(concatenate_prototypes)
                .view(batch_size, self.layers, self.directions, self.d_model)
                .permute(1, 2, 0, 3).contiguous() # (L, Dir, B, D)
                .view(self.layers * self.directions, batch_size, self.d_model)
        )
        lang = []

        # Create and add SOS token
        sos_onehot = torch.zeros(batch_size, 1, self.vocabulary).to(prototypes[0].device)
        sos_onehot[:, 0, data.language.SOS_IDX] = 1.0
        lang.append(sos_onehot)

        inputs = sos_onehot # Shape: (B, 1, V)
        inputs = inputs @ self.token_embedding.weight  # Shape: (B, 1, D)

        # Main sampling loop (fixed length of max_len - 2)
        for i in range(self.max_len - 2):

            # Input is (B, 1, D), Output is (B, 1, H)
            outputs, states = self.gru(inputs, states)

            # outputs = outputs.squeeze(1) # Shape: (B, H)
            outputs = self.outputs2vocab(outputs[:, -1, :]) # Shape: (B, V)

            outputs = F.gumbel_softmax(outputs, tau=self.softmax_temp, hard=False)

            if self.uniform_weight != 0.0:
                uniform_outputs = torch.full_like(
                    outputs,
                    1 / outputs.shape[-1]
                )
                outputs = (
                    (1 - self.uniform_weight) * outputs
                    +
                    self.uniform_weight * uniform_outputs
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

            lang.append(predicted_onehot.unsqueeze(1)) # (B, 1, V)

            inputs = (predicted_onehot.unsqueeze(1)) @ self.token_embedding.weight # (B, 1, D)

        # Add final EOS token
        eos_onehot = torch.zeros(batch_size, 1, self.vocabulary).to(prototypes[0].device)
        eos_onehot[:, 0, data.language.EOS_IDX] = 1.0
        lang.append(eos_onehot)

        # Concatenate along the sequence dim (1)
        lang_tensor = torch.cat(lang, 1) # (B, max_len, V)

        return lang_tensor

    def reset_parameters(self):
        self.init_h.reset_parameters()
        self.gru.reset_parameters()
        self.outputs2vocab.reset_parameters()
        self.token_embedding.reset_parameters()


class SenderTransformerLM(nn.Module):
    def __init__(
        self,
        referent_embedding_size,
        token_embedding_size,
        d_model,
        vocabulary,
        max_len,
        softmax_temp,
        uniform_weight,
        dropout,
        layers,
        bidirectional,
        heads = 4,
        bos_tokens = 16
    ):
        super().__init__()
        self.referent_embedding_size = referent_embedding_size
        self.token_embedding_size = token_embedding_size
        self.d_model = d_model
        self.vocabulary = vocabulary
        self.max_len = max_len
        self.softmax_temp = softmax_temp
        self.uniform_weight = uniform_weight
        self.dropout = dropout
        self.layers = layers
        self.bidirectional = bidirectional
        self.heads = heads
        self.bos_tokens = bos_tokens

        if self.referent_embedding_size != self.token_embedding_size:
            raise NotImplementedError(
                "`referent_embedding_size` and `token_embedding_size` must "
                "be the same for Transformer-based speaker models!"
            )

        if int((self.d_model / self.heads) / self.bos_tokens) < 3:
            warnings.warn(
                "Fewer than 3 head dimensions per BOS token may be suboptimal."
            )

        if max_len < 3:
            raise ValueError("max_len must be at least 3 (due to SOS and EOS tokens)")

        self.content_length = self.max_len - 2

        self.query = nn.Parameter(
            torch.empty(self.bos_tokens, self.d_model)
        )
        nn.init.normal_(self.query, mean=0.0, std=1.0)

        self.cross_attention = broccoli.transformer.MHAttention(
            self.d_model,
            self.heads,
            dropout=self.dropout,
            causal=False, # Whole image informs whole initial message
            seq_len=self.content_length,
            scaling="d",
        )

        self.transformer = broccoli.transformer.TransformerEncoder(
            self.bos_tokens + self.content_length,
            self.d_model,
            self.layers,
            self.heads,
            absolute_position_embedding=False, # We manually add this in
            relative_position_embedding=True,
            source_size=(self.content_length,),
            mlp_ratio=2,
            activation=broccoli.activation.SwiGLU,
            activation_kwargs=None,
            mlp_dropout=0.,
            msa_dropout=0.,
            stochastic_depth=0.2,
            causal = not self.bidirectional,
            bos_tokens=0, # BOS tokens are used differently here
            return_bos_tokens=False,
        )

        self.outputs2vocab = nn.Linear(d_model, vocabulary)

        self.position_embedding = nn.Embedding(self.content_length, self.d_model)

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
            self.bos_tokens,
            self.d_model
        )

        bos_tokens = self.cross_attention(
            query, stack_prototypes, stack_prototypes
        ) # (batch, self.bos_tokens, self.d_model)

        positions = torch.arange(
            0,
            self.content_length,
            dtype=torch.long,
            device=prototypes[0].device
        ).unsqueeze(0) # shape (1, t)

        position_embeddings = self.position_embedding(positions) # (1, self.content_length, self.d_model)

        position_embeddings = position_embeddings.expand(
            batch_size,
            self.content_length,
            self.d_model
        )

        input_sequences = torch.cat([bos_tokens, position_embeddings], dim=1)

        outputs = self.transformer(input_sequences)[:, self.bos_tokens:, :] # Strip out BOS tokens
        logits = self.outputs2vocab(outputs)
        outputs = F.gumbel_softmax(logits, tau=self.softmax_temp, hard=False)

        if self.uniform_weight != 0.0:
            uniform_outputs = torch.full_like(
                outputs,
                1 / outputs.shape[-1]
            )
            outputs = (
                (1 - self.uniform_weight) * outputs
                +
                self.uniform_weight * uniform_outputs
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

        sos_onehot = torch.zeros(batch_size, 1, self.vocabulary, device=device)
        sos_onehot[:, 0, data.language.SOS_IDX] = 1.0
        eos_onehot = torch.zeros(batch_size, 1, self.vocabulary, device=device)
        eos_onehot[:, 0, data.language.EOS_IDX] = 1.0

        onehot = torch.cat([sos_onehot, onehot_content, eos_onehot], dim=1)

        return onehot # (batch, max_len, vocabulary)

    def reset_parameters(self):
        nn.init.normal_(self.query, mean=0.0, std=1.0)
        self.cross_attention.reset_parameters()
        self.transformer.reset_parameters()
        self.outputs2vocab.reset_parameters()


class SenderGPTLM(nn.Module):
    def __init__(
        self,
        referent_embedding_size,
        token_embedding_size,
        d_model,
        vocabulary,
        max_len,
        softmax_temp,
        uniform_weight,
        dropout,
        layers,
        bidirectional,
        heads = 4,
        bos_tokens = 16
    ):
        super().__init__()
        self.referent_embedding_size = referent_embedding_size
        self.token_embedding_size = token_embedding_size
        self.d_model = d_model
        self.vocabulary = vocabulary
        self.max_len = max_len
        self.softmax_temp = softmax_temp
        self.uniform_weight = uniform_weight
        self.dropout = dropout
        self.layers = layers
        self.bidirectional = bidirectional
        self.heads = heads
        self.bos_tokens = bos_tokens

        if self.referent_embedding_size != self.token_embedding_size:
            raise NotImplementedError(
                "`referent_embedding_size` and `token_embedding_size` must "
                "be the same for Transformer-based speaker models!"
            )

        if int((self.d_model / self.heads) / self.bos_tokens) < 3:
            warnings.warn(
                "Fewer than 3 head dimensions per BOS token may be suboptimal."
            )

        if max_len < 3:
            raise ValueError("max_len must be at least 3 (due to SOS and EOS tokens)")

        self.content_length = self.max_len - 2

        self.query = nn.Parameter(
            torch.empty(self.bos_tokens, self.d_model)
        )
        nn.init.normal_(self.query, mean=0.0, std=1.0)

        self.cross_attention = broccoli.transformer.MHAttention(
            self.d_model,
            self.heads,
            dropout=self.dropout,
            causal=False, # Whole image informs whole initial message
            seq_len=self.content_length,
            scaling="d",
        )

        self.transformer = broccoli.transformer.TransformerEncoder(
            self.bos_tokens + self.content_length,
            self.d_model,
            self.layers,
            self.heads,
            absolute_position_embedding=False, # We manually add this in
            relative_position_embedding=True,
            source_size=(self.content_length,),
            mlp_ratio=2,
            activation=broccoli.activation.SwiGLU,
            activation_kwargs=None,
            mlp_dropout=0.,
            msa_dropout=0.,
            stochastic_depth=0.2,
            causal = not self.bidirectional,
            bos_tokens=0, # BOS tokens are used differently here
            return_bos_tokens=False,
        )

        self.outputs2vocab = nn.Linear(d_model, vocabulary)

        self.position_embedding = nn.Embedding(self.content_length, self.d_model)
        
        self.token_embedding = nn.Embedding(vocabulary, token_embedding_size)

    def forward(
        self,
        prototypes,
        **kwargs
    ):
        batch_size = prototypes[0].size(0)
        device = prototypes[0].device

        sos_onehot = torch.zeros(batch_size, 1, self.vocabulary, device=device)
        sos_onehot[:, 0, data.language.SOS_IDX] = 1.0
        eos_onehot = torch.zeros(batch_size, 1, self.vocabulary, device=device)
        eos_onehot[:, 0, data.language.EOS_IDX] = 1.0

        stack_prototypes = torch.stack(prototypes, 1)

        query = self.query.unsqueeze(0).expand(
            batch_size,
            self.bos_tokens,
            self.d_model
        )

        bos_tokens = self.cross_attention(
            query, stack_prototypes, stack_prototypes
        ) # (batch, self.bos_tokens, self.d_model)

        positions = torch.arange(
            0,
            self.content_length,
            dtype=torch.long,
            device=prototypes[0].device
        ).unsqueeze(0) # shape (1, t)

        position_embeddings = self.position_embedding(positions) # (1, self.content_length, self.d_model)

        position_embeddings = position_embeddings.expand(
            batch_size,
            self.content_length,
            self.d_model
        )

        input_sequences = torch.cat([bos_tokens, position_embeddings], dim=1)

        spoken_so_far = torch.empty(
            (batch_size, 0, self.d_model),
            dtype=position_embeddings.dtype,
            device=position_embeddings.device
        )

        # Main sampling loop (fixed length of max_len - 2)
        for i in range(self.content_length):

            input_sequences = torch.cat(
                [
                    bos_tokens,
                    spoken_so_far[:i],
                    position_embeddings[:, i:, :]
                ],
                dim=1
            )

            outputs = self.transformer(input_sequences)[:, self.bos_tokens:, :] # Strip out BOS tokens
            logits = self.outputs2vocab(outputs)
            outputs = F.gumbel_softmax(logits, tau=self.softmax_temp, hard=False)

            if self.uniform_weight != 0.0:
                uniform_outputs = torch.full_like(
                    outputs,
                    1 / outputs.shape[-1]
                )
                outputs = (
                    (1 - self.uniform_weight) * outputs
                    +
                    self.uniform_weight * uniform_outputs
                )
                    
            onehot_content = (
                torch.zeros_like(outputs)
                    .scatter_(
                        -1,
                        torch.argmax(outputs, dim=-1).unsqueeze(-1),
                        1.
                    )
                )

            onehot_content = onehot_content - outputs.detach() + outputs # (B, self.content_length, V)

            spoken_so_far = onehot_content[:, :i + 1, :] @ self.token_embedding.weight # (B, self.content_length, self.d_model)

            current_message = torch.cat([sos_onehot, onehot_content, eos_onehot], dim=1)

            if self.bidirectional:
                break

        return current_message

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
        if samples.size(1) % 2 != 0:
            raise NotImplementedError(
                "The prototyper must be passed an even number of samples, "
                "the first n / 2 should be positive and the rest negative."
            )
            
        midp = targets.shape[1] // 2

        if not ((targets[:, :midp] == 1.0).all() and (targets[:, midp:] == 0.0).all()):
            raise NotImplementedError(
                "The prototyper must be passed an even number of samples, "
                "the first n / 2 should be positive and the rest negative."
            )
        
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
        if hasattr(self.feat_model, 'reset_parameters'):
            self.feat_model.reset_parameters()
        self.prototyper.reset_parameters()
        self.language_model.reset_parameters()