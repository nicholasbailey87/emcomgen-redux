"""
Speaker models. This includes speakers with a GRU-based language model as
    originally presented in "Emergent Communication of Generalizations"
    (https://arxiv.org/abs/2106.02668) and speakers with causal or non-causal
    Transformer language models. The intention is to show that
    Transformer-based speakers can be just as successful in tasks and show
    equal or greater compositionality.
"""

import warnings
import math

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

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
        **kwargs
    ):
        super().__init__()
        self.referent_embedding_size = referent_embedding_size
        self.token_embedding_size = kwargs["token_embedding_size"]
        self.d_model = kwargs["d_model"]
        self.vocabulary = kwargs["vocabulary"]
        self.message_length = kwargs["message_length"]
        self.softmax_temperature = kwargs["softmax_temperature"]
        self.uniform_weight = kwargs["uniform_weight"]
        self.dropout = kwargs["dropout"]
        self.layers = kwargs["layers"]
        self.bidirectional = kwargs["bidirectional"]
        self.directions = 2 if self.bidirectional else 1
        
        self.gru = nn.GRU(
            self.token_embedding_size,
            self.d_model,
            num_layers=self.layers,
            bias=True,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )

        self.outputs2vocab = nn.Linear(
            self.d_model * self.directions,
            self.vocabulary + 2 # +2 for SOS and EOS
        )

        self.init_h = nn.Linear(
            2 * referent_embedding_size, 
            self.layers * self.directions * self.d_model
        )
        
        self.token_embedding = nn.Embedding(
            self.vocabulary + 2, # +2 for SOS and EOS
            self.token_embedding_size
        )

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
        sos_onehot = torch.zeros(batch_size, 1, self.vocabulary + 2).to(prototypes[0].device)
        sos_onehot[:, 0, data.language.SOS_IDX] = 1.0
        lang.append(sos_onehot)

        inputs = sos_onehot # Shape: (B, 1, V)
        inputs = inputs @ self.token_embedding.weight  # Shape: (B, 1, D)

        # Main sampling loop (fixed length of message_length - 2)
        for i in range(self.message_length - 2):

            # Input is (B, 1, D), Output is (B, 1, H)
            outputs, states = self.gru(inputs, states)

            # outputs = outputs.squeeze(1) # Shape: (B, H)
            outputs = self.outputs2vocab(outputs[:, -1, :]) # Shape: (B, V)

            outputs = F.gumbel_softmax(outputs, tau=self.softmax_temperature, hard=False)

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
        eos_onehot = torch.zeros(batch_size, 1, self.vocabulary + 2).to(prototypes[0].device)
        eos_onehot[:, 0, data.language.EOS_IDX] = 1.0
        lang.append(eos_onehot)

        # Concatenate along the sequence dim (1)
        lang_tensor = torch.cat(lang, 1) # (B, message_length, V)

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
        **kwargs
    ):
        """
        The idea here is that we create a model that can accept some prototype
            embeddings of referents and output a message.
        
        We allow the model to learn a fixed query tensor, of size
            (1, message_lengthgth - 2, d_model)
        
            We use "message_length - 2" as we save space for SOS and EOS tokens.

        We use cross-attention with this query and the sequence of prototype
            embeddings to produce an initial sequence for input to a Transformer.
        
        The Transformer produces a message-length sequence of embedded tokens,
            which we project into the vocabulary space with an nn.Linear
            module.

        We standardise the logits and create a vector of gumbel noise, which we
            also standardise. We linearly interpolate between these two vectors
            for each token in each sample in the batch, like

            `confidence * normalised_logits + (1-confidence) * noise`
        
            where `confidence` is determined by the entropy of the logits in
            the following way:

            `c = H(normalised_logits) / log_2(vocabulary)`

        The gumbel noise helps the model to explore new vocabulary, and by
            using the method described above we encourage more exploration
            when the model is less sure what word to use in a given context.
        
        The evolving preferences of the listener agent(s) is ultimately what
            dictates the amount of positive feedback for each message, which
            introduces a mitigation for the "confident idiot" issue where a
            consistently very confident but very wrong word choice gets
            little exploration: a consistent word choice in a given context
            simply becomes part of the language.
        
        We prefer softmax temperature higher than 1 (e.g. 16) when using
            straight-through Gumbel softmax, for the reasons described here:
            https://arxiv.org/abs/2502.20604
        """
        super().__init__()
        self.referent_embedding_size = referent_embedding_size
        self.token_embedding_size = kwargs["token_embedding_size"]
        self.d_model = kwargs["d_model"]
        self.vocabulary = kwargs["vocabulary"]
        self.max_entropy = math.log(self.vocabulary)
        self.message_length = kwargs["message_length"]
        self.softmax_temperature = kwargs["softmax_temperature"]
        self.confidence_based_exploration = kwargs["confidence_based_exploration"]
        self.uniform_weight = kwargs["uniform_weight"]
        self.dropout = kwargs["dropout"]
        self.layers = kwargs["layers"]
        self.bidirectional = kwargs["bidirectional"]
        self.heads = kwargs["heads"]
        self.utility_tokens = kwargs["utility_tokens"]

        if self.referent_embedding_size != self.token_embedding_size:
            raise NotImplementedError(
                "`referent_embedding_size` and `token_embedding_size` must "
                "be the same for Transformer-based speaker models!"
            )

        if int((self.d_model / self.heads) / self.utility_tokens) < 3:
            warnings.warn(
                "Fewer than 3 head dimensions per utility token may be suboptimal."
            )

        if self.message_length < 3:
            raise ValueError(
                "message_length must be at least 3 (due to SOS and EOS tokens)"
            )

        self.content_length = self.message_length - 2

        self.query = nn.Parameter(
            torch.empty(self.content_length, self.d_model)
        )

        self.cross_attention = broccoli.transformer.MHAttention(
            self.d_model,
            self.heads,
            dropout=self.dropout,
            causal=False, # Whole image informs whole initial message
            seq_len=self.content_length,
            scaling="d",
        )

        self.transformer = broccoli.transformer.TransformerEncoder(
            self.utility_tokens + self.content_length,
            self.d_model,
            self.layers,
            self.heads,
            absolute_position_embedding=True,
            relative_position_embedding=True,
            source_size=(self.content_length,),
            mlp_ratio=2,
            activation=broccoli.activation.SwiGLU,
            activation_kwargs=None,
            mlp_dropout=0.,
            msa_dropout=0.,
            stochastic_depth=0.2,
            causal = not self.bidirectional,
            utility_tokens=self.utility_tokens,
            return_utility_tokens=False,
        )

        self.outputs2vocab = nn.Linear(
            self.d_model,
            self.vocabulary + 2 # +2 for SOS and EOS
        )

        self.reset_parameters()

    def get_noise_coefficients(self, logits):
        """
        Calculates the per-token noise coefficient (0.0 to 1.0) for every token
            in a batch of generated messages. Returns shape: (B, L, 1).
        """
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            token_entropy = -(probs * log_probs).sum(dim=-1)
            norm_entropy = token_entropy / self.max_entropy
            norm_entropy = torch.clamp(norm_entropy, 0.0, 1.0)
            return norm_entropy.unsqueeze(-1)

    def forward(
        self,
        prototypes,
        **kwargs
    ):
        batch_size = prototypes[0].size(0)
        device = prototypes[0].device
        finfo = torch.finfo(prototypes[0].dtype)
        eps = finfo.eps

        stack_prototypes = torch.stack(prototypes, 1) # To sequence

        query = self.query.unsqueeze(0).expand(
            batch_size,
            self.content_length,
            self.d_model
        )

        initial_sequence = self.cross_attention(
            query, stack_prototypes, stack_prototypes
        ) # (batch, self.content_length, self.d_model)

        outputs = self.transformer(initial_sequence)

        logits = self.outputs2vocab(outputs)

        uniform_distribution = torch.clamp(
            torch.rand_like(logits),
            min=eps,
            max=1.0 - eps
        )

        if self.confidence_based_exploration:
            noise_coefficients = self.get_noise_coefficients(logits)

            logits_mean = logits.mean(dim=-1, keepdim=True)
            logits_std = logits.std(dim=-1, keepdim=True)
            logits_norm = (logits - logits_mean) / (logits_std + eps)

            gumbel_distribution = -torch.log(-torch.log(uniform_distribution))
            # The Gumbel noise should have approximately these mean and std:
            gumbel_norm = (gumbel_distribution - 0.58) / 1.28

            logits = (
                noise_coefficients * gumbel_norm
                +
                (1 - noise_coefficients) * logits_norm
            )
        else: # Just add Gumbel noise
            # This looks like subtraction but it should be correct!
            logits = logits - torch.log(-torch.log(uniform_distribution))
        
        logits /= self.softmax_temperature
        outputs = F.softmax(logits, dim=-1)

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

        sos_onehot = torch.zeros(batch_size, 1, self.vocabulary + 2, device=device)
        sos_onehot[:, 0, data.language.SOS_IDX] = 1.0
        eos_onehot = torch.zeros(batch_size, 1, self.vocabulary + 2, device=device)
        eos_onehot[:, 0, data.language.EOS_IDX] = 1.0

        onehot = torch.cat([sos_onehot, onehot_content, eos_onehot], dim=1)

        return onehot # (batch, message_length, vocabulary)

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
        vision_dropout: float= 0.5
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
        self.vision_dropout = nn.Dropout(p=vision_dropout)

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
        embedded_samples = self.vision_dropout(self.feat_model(flat_samples))
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