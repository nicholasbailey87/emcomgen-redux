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

import einops

import data
import data.language

import broccoli

def batch_norm_logits(module: nn.BatchNorm1d, logits: torch.Tensor) -> torch.Tensor:
    """
    Applies nn.BatchNorm1d to vocabulary logits that are arranged in a
        sequence like (batch, seq, vocabulary)
    
    Args:
        module: The nn.BatchNorm1d module
        logits: Tensor of shape (Batch, Length, Vocab)
        
    Returns:
        Tensor of shape (Batch, Length, Vocab)
    """
    logits = module(einops.rearrange(logits, 'b l c -> b c l'))
    return einops.rearrange(logits, 'b c l -> b l c')

def flatten_logit_distribution(
    logits: torch.Tensor,
    uniform_weight: float
) -> torch.Tensor:
    """
    Returns a weighted average of 

    Args:
        logits: some provided unnormalised log probabilities
        uniform_weight: the relative weight to give the uniform distribution
            when mixing it in to the provided logits
    
    Returns:
        A torch.Tensor of logits where the absolute differences between
            logits is reduced - i..e. a less "certain" distribution
    """
    normalised_logits = F.log_softmax(logits, dim=-1)
    # Make a uniform distribution, but in the log space
    uniform_log_probs = torch.full_like(
        normalised_logits,
        -np.log(logits.shape[-1])
    )
    
    # Mix the log distributions, like log( w * uniform + (1-w) * model )
    combined_logits = torch.stack(
        [
            uniform_log_probs + np.log(uniform_weight),
            logits + np.log(1 - uniform_weight),
        ],
        dim=-1,
    )

    return torch.logsumexp(combined_logits, dim=-1)


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
        self.exploration_temperature = kwargs["exploration_temperature"]
        self.uniform_weight = kwargs["uniform_weight"]
        self.batch_norm_logits = kwargs["batch_norm_logits"]
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
            self.vocabulary + 4 # +4 for PAD, SOS, EOS, UNK
        )

        if self.batch_norm_logits:
            self.batch_norm = nn.BatchNorm1d(self.vocabulary + 4) # +4 for PAD, SOS, EOS, UNK
        else:
            self.batch_norm = None

        self.init_h = nn.Linear(
            2 * referent_embedding_size, 
            self.layers * self.directions * self.d_model
        )
        
        self.token_embedding = nn.Embedding(
            self.vocabulary + 4, # +4 for PAD, SOS, EOS, UNK
            self.token_embedding_size
        )

        self.reset_parameters()

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
        device = prototypes[0].device

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
        sos_onehot = torch.zeros(
            batch_size,
            1,
            self.vocabulary + 4, # +4 for PAD, SOS, EOS, UNK
            device=device
        )  # Shape: (B, 1, V)
        sos_onehot[:, 0, data.language.SOS_IDX] = 1.0
        lang.append(sos_onehot)

        gru_in = sos_onehot @ self.token_embedding.weight  # Shape: (B, 1, D)

        # Main sampling loop (fixed length of message_length - 2)
        for i in range(self.message_length - 2):

            gru_out, states = self.gru(gru_in, states)
            
            logits = self.outputs2vocab(gru_out[:, -1, :]) # Shape: (B, V)

            if self.batch_norm_logits:
                # This must come before the uniform weight mixing
                #     as it would otherwise mess up the distribution
                logits = self.batch_norm(logits)
            
            if self.uniform_weight > 0.0:
                logits = flatten_logit_distribution(logits, self.uniform_weight)

            logits = logits / self.exploration_temperature
            
            # Remove probability mass from reserved tokens
            # Probability mass there should atrophy anyway as it won't have gradient(?)
            logits[:, :4] = -float('inf')

            # 5. Gumbel-Softmax (hard=True)
            # This handles the noise addition + argmax + straight-through gradient
            predicted_onehot = F.gumbel_softmax(
                logits,
                tau=self.softmax_temperature,
                hard=True,
                dim=-1
            )

            # 6. Prepare next input
            lang.append(predicted_onehot.unsqueeze(1))
            gru_in = (predicted_onehot.unsqueeze(1)) @ self.token_embedding.weight # (B, 1, D)

        # Add final EOS token
        eos_onehot = torch.zeros(batch_size, 1, self.vocabulary + 2, device=device)
        eos_onehot[:, 0, data.language.EOS_IDX] = 1.0
        lang.append(eos_onehot)

        # Concatenate
        lang_tensor = torch.cat(lang, 1) 

        return lang_tensor

    def reset_parameters(self):
        self.init_h.reset_parameters()
        self.gru.reset_parameters()
        self.outputs2vocab.reset_parameters()
        self.token_embedding.reset_parameters()
        if self.batch_norm is not None:
            self.batch_norm.reset_parameters()


class SenderTransformerLM(nn.Module):
    def __init__(
        self,
        referent_embedding_size,
        **kwargs
    ):
        """
        ...
        
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
        self.exploration_temperature = kwargs["exploration_temperature"]
        self.uniform_weight = kwargs["uniform_weight"]
        self.batch_norm_logits = kwargs["batch_norm_logits"]
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

        self.query_layer_norm = nn.LayerNorm(self.d_model)
        self.referent_layer_norm = nn.LayerNorm(self.d_model)

        self.cross_attention = broccoli.transformer.MHAttention(
            self.d_model,
            self.heads,
            dropout=self.dropout,
            causal=False, # Whole image informs whole initial message
            seq_len=self.content_length,
            scaling="d",
        )

        self.transformer = broccoli.transformer.TransformerEncoder(
            self.content_length,
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
            pre_norm=True,
            post_norm=True,
            normformer=True,
            msa_scaling="d",
        )

        self.outputs2vocab = nn.Linear(
            self.d_model,
            self.vocabulary + 4 # +4 for PAD, SOS, EOS, UNK
        )

        if self.batch_norm_logits:
            self.batch_norm = nn.BatchNorm1d(self.vocabulary + 4) # +4 for PAD, SOS, EOS, UNK
        else:
            self.batch_norm = None

        self.reset_parameters()

    def forward(
        self,
        prototypes,
        **kwargs
    ):
        batch_size = prototypes[0].size(0)
        device = prototypes[0].device

        stack_prototypes = torch.stack(prototypes, 1) # To sequence

        normed_prototypes = self.referent_layer_norm(stack_prototypes)

        query = self.query.unsqueeze(0).expand(
            batch_size,
            self.content_length,
            self.d_model
        )

        normed_query = self.query_layer_norm(query)

        initial_sequence = self.cross_attention(
            normed_query, normed_prototypes, normed_prototypes
        ) # (batch, self.content_length, self.d_model)

        outputs = self.transformer(initial_sequence)

        logits = self.outputs2vocab(outputs)

        if self.batch_norm_logits:
            # This must come before the uniform weight mixing
            #     as it would otherwise mess up the distribution
            logits = batch_norm_logits(self.batch_norm, logits)
            
        if self.uniform_weight > 0.0:
            logits = flatten_logit_distribution(logits, self.uniform_weight)

        logits = logits / self.exploration_temperature
        
        # Remove probability mass from reserved tokens
        # Probability mass there should atrophy anyway as it won't have gradient(?)
        logits[:, :, :4] = -float('inf')

        onehot_content = F.gumbel_softmax(
            logits,
            tau=self.softmax_temperature,
            hard=True,
            dim=-1
        )

        sos_onehot = torch.zeros(batch_size, 1, self.vocabulary + 4, device=device)
        sos_onehot[:, 0, data.language.SOS_IDX] = 1.0
        eos_onehot = torch.zeros(batch_size, 1, self.vocabulary + 4, device=device)
        eos_onehot[:, 0, data.language.EOS_IDX] = 1.0

        onehot = torch.cat([sos_onehot, onehot_content, eos_onehot], dim=1)

        return onehot # (batch, message_length, vocabulary)

    def reset_parameters(self):
        nn.init.normal_(self.query, mean=0.0, std=1.0)
        self.query_layer_norm.reset_parameters()
        self.referent_layer_norm.reset_parameters()
        self.cross_attention.reset_parameters()
        self.transformer.reset_parameters()
        self.outputs2vocab.reset_parameters()
        if self.batch_norm is not None:
            self.batch_norm.reset_parameters()


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