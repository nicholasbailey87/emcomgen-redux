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

import einops

import data
import data.language

import broccoli


def trim_messages(token_id_rows):
    """
    Turn rows of decoded token ids into ragged content-token sequences: drop the
    leading SOS and truncate at the first EOS. The language model masks the
    reserved tokens (PAD/SOS/EOS/UNK) mid-sequence, so the surviving tokens are
    all real symbols. Accepts anything iterable-of-iterables (e.g. a CPU tensor
    via ``.tolist()``); returns a list of python int lists.
    """
    sos, eos = data.language.SOS_IDX, data.language.EOS_IDX
    trimmed = []
    for row in token_id_rows:
        toks = []
        for t in row:
            t = int(t)
            if t == sos:
                continue
            if t == eos:
                break
            toks.append(t)
        trimmed.append(toks)
    return trimmed


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

def mask_reserved_tokens(logits: torch.Tensor) -> torch.Tensor:
    """
    Set the four reserved tokens (PAD/SOS/EOS/UNK) to -inf so they can never be
        emitted mid-message. SOS and EOS are attached by the caller instead, so
        messages are fixed-length.

    Out of place, because this now runs directly on the output of the vocabulary
        projection (or of batch norm), and writing -inf into that in place would
        be modifying a tensor autograd still needs.

    Runs before the exploration noise so that the uniform mixture is spread over
        the emittable tokens only — see `flatten_logit_distribution`.

    Args:
        logits: (..., vocabulary + 4), reserved tokens first

    Returns:
        A tensor of the same shape with the reserved positions set to -inf
    """
    reserved = torch.zeros_like(logits, dtype=torch.bool)
    reserved[..., :4] = True
    return logits.masked_fill(reserved, -float("inf"))


def flatten_logit_distribution(
    logits: torch.Tensor,
    uniform_weight: float
) -> torch.Tensor:
    """
    Returns a weighted average of

    The uniform component is spread over the emittable tokens only, i.e. those
        not already masked to -inf by `mask_reserved_tokens`. Spreading it over
        all `vocabulary + 4` slots and masking afterwards would throw away the
        4/(V+4) of it that landed on reserved tokens, so a nominal weight of
        0.1 would deliver 0.078. Masked positions stay masked.

    Args:
        logits: some provided unnormalised log probabilities
        uniform_weight: the relative weight to give the uniform distribution
            when mixing it in to the provided logits

    Returns:
        A torch.Tensor of logits where the absolute differences between
            logits is reduced - i..e. a less "certain" distribution
    """
    emittable = torch.isfinite(logits)
    normalised_logits = F.log_softmax(logits, dim=-1)

    # Make a uniform distribution, but in the log space
    uniform_log_probs = -torch.log(
        emittable.sum(-1, keepdim=True).to(logits.dtype)
    ).expand_as(logits)

    # Mix the log distributions, like log( w * uniform + (1-w) * model ).
    # Masked entries are -inf in both components, and `logsumexp` of two -inf
    #     backpropagates NaN, so they are mixed as a finite placeholder and the
    #     mask is restored afterwards. `torch.where` routes the gradient to the
    #     selected branch only, so the placeholder never reaches the speaker.
    placeholder = torch.zeros_like(logits)
    combined_logits = torch.stack(
        [
            torch.where(emittable, uniform_log_probs, placeholder)
            + np.log(uniform_weight),
            torch.where(emittable, normalised_logits, placeholder)
            + np.log(1 - uniform_weight),
        ],
        dim=-1,
    )

    return torch.where(
        emittable, torch.logsumexp(combined_logits, dim=-1), logits
    )


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
        self.tau = kwargs["tau"]  # Gumbel-softmax tau, as in jayelm
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

    def decode(
        self,
        prototypes,
        return_states=True
    ):
        """
        Run the decoding loop once, returning both the message and the symbol
            embeddings that produced it.

        Unlike `SenderTransformerLM`, whose embeddings are a function of the
            prototypes alone, this speaker is autoregressive: each embedding
            depends on the symbols sampled before it, so embeddings and
            message only correspond when they come from the *same* call.
            `forward` discards the embeddings, so any analysis needing the two
            to be paired must call this directly (as `Sender.speak` does).

        Returns:
            lang_tensor: (batch, message_length, vocabulary + 4)
            symbol_embeddings: (batch, message_length - 2, d_model *
                directions), one dense contextual vector per content symbol,
                taken before the vocabulary projection and sampling, or None
                when `return_states` is False. Stacking them is a copy the
                training path has no use for, so `forward` opts out.
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
        symbol_embeddings = []

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

            step_embedding = gru_out[:, -1, :] # Shape: (B, D * Dir)
            symbol_embeddings.append(step_embedding)

            logits = self.outputs2vocab(step_embedding) # Shape: (B, V)

            if self.batch_norm_logits:
                # This must come before the uniform weight mixing
                #     as it would otherwise mess up the distribution
                logits = self.batch_norm(logits)

            # Masking comes first so that the uniform mixture below is spread
            #     over the emittable tokens only.
            logits = mask_reserved_tokens(logits)

            # Exploration is a training-time device only, so that the eval
            #     passes measure the learned policy rather than a deliberately
            #     noised one. Mirrors jayelm's emergent-generalization, which
            #     zeroes `uniform_weight` and resets `softmax_temp` to 1.0
            #     whenever the split is not `train`.
            if self.training:
                if self.uniform_weight > 0.0:
                    logits = flatten_logit_distribution(logits, self.uniform_weight)

                logits = logits / self.exploration_temperature

            # 5. Gumbel-Softmax (hard=True)
            # This handles the noise addition + argmax + straight-through gradient.
            # Note `tau` rescales the *soft* sample only: the hard forward sample
            #     is an argmax and so is invariant to it. It is a gradient knob,
            #     not an exploration knob — `uniform_weight` is the latter.
            predicted_onehot = F.gumbel_softmax(
                logits,
                tau=self.tau,
                hard=True,
                dim=-1
            )

            # 6. Prepare next input
            lang.append(predicted_onehot.unsqueeze(1))
            gru_in = (predicted_onehot.unsqueeze(1)) @ self.token_embedding.weight # (B, 1, D)

        # Add final EOS token
        eos_onehot = torch.zeros(batch_size, 1, self.vocabulary + 4, device=device)
        eos_onehot[:, 0, data.language.EOS_IDX] = 1.0
        lang.append(eos_onehot)

        # Concatenate
        lang_tensor = torch.cat(lang, 1)

        return (
            lang_tensor,
            torch.stack(symbol_embeddings, 1) if return_states else None
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
        return self.decode(prototypes, return_states=False)[0]

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
        self.message_length = kwargs["message_length"]
        self.tau = kwargs["tau"]  # Gumbel-softmax tau, as in jayelm
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

        if (
            self.utility_tokens
            and (int((self.d_model / self.heads) / self.utility_tokens) < 3)
        ):
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
            ff_ratio=2,
            activation=broccoli.activation.SwiGLU,
            activation_kwargs=None,
            ff_dropout=0.,
            msa_dropout=0.,
            stochastic_depth=0.2,
            causal = not self.bidirectional,
            bos_tokens=self.utility_tokens,
            return_bos_tokens=False,
            pre_norm=False,
            post_norm=True,
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

    def embeddings(
        self,
        prototypes
    ):
        batch_size = prototypes[0].size(0)

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

        return self.transformer(initial_sequence)

    def decode(
        self,
        prototypes
    ):
        """
        Produce a message and the symbol embeddings that produced it, in a
            single pass. Mirrors `SenderGRULM.decode`.

        This speaker is not autoregressive, so unlike the GRU the embeddings
            here do not depend on which symbols were sampled. This method
            exists so that callers can treat the two speakers identically.

        Returns:
            onehot: (batch, message_length, vocabulary + 4)
            symbol_embeddings: (batch, message_length - 2, d_model), one
                dense contextual vector per content symbol, taken before the
                vocabulary projection and sampling
        """
        batch_size = prototypes[0].size(0)
        device = prototypes[0].device

        symbol_embeddings = self.embeddings(prototypes)

        logits = self.outputs2vocab(symbol_embeddings)

        if self.batch_norm_logits:
            # This must come before the uniform weight mixing
            #     as it would otherwise mess up the distribution
            logits = batch_norm_logits(self.batch_norm, logits)

        # Mask first, then explore, training-time only — as in
        #     `SenderGRULM.decode`, see the notes there.
        logits = mask_reserved_tokens(logits)

        if self.training:
            if self.uniform_weight > 0.0:
                logits = flatten_logit_distribution(logits, self.uniform_weight)

            logits = logits / self.exploration_temperature

        onehot_content = F.gumbel_softmax(
            logits,
            tau=self.tau,
            hard=True,
            dim=-1
        )

        sos_onehot = torch.zeros(batch_size, 1, self.vocabulary + 4, device=device)
        sos_onehot[:, 0, data.language.SOS_IDX] = 1.0
        eos_onehot = torch.zeros(batch_size, 1, self.vocabulary + 4, device=device)
        eos_onehot[:, 0, data.language.EOS_IDX] = 1.0

        onehot = torch.cat([sos_onehot, onehot_content, eos_onehot], dim=1)

        return onehot, symbol_embeddings

    def forward(
        self,
        prototypes,
        **kwargs
    ):
        return self.decode(prototypes)[0] # (batch, message_length, vocabulary)

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
        vision_dropout: float= 0.5,
        prototype_dropout: float= 0.5
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
            vision_dropout: Dropout probability between the `feat_model` and the
                `prototyper`, i.e. on per-image embeddings, before pooling
            prototype_dropout: Dropout probability between the `prototyper` and
                the `language_model`, i.e. on the pooled concept vectors. This
                is where jayelm's single `--dropout` sits (on the speaker side);
                dropping features before the pool is much weaker, since the
                average over n/2 examples largely restores them.
        """
        super().__init__()
        self.feat_model = feat_model
        self.feat_size = feat_model.final_feat_dim
        self.prototyper = prototyper
        self.language_model = language_model
        self.vision_dropout = nn.Dropout(p=vision_dropout)
        self.prototype_dropout = nn.Dropout(p=prototype_dropout)

    def embed_images(self, samples):
        """
        Embed all the referent images in a batch.
        Input size is (batch, referents, image dim 1, image dim 2, etc.)
        We reshape to (batch_size * n_obj, *rest) to get a batch of images, then
            send them through the computer vision model, then reshape back to the
            shape needed for the task.
        """
        batch_size = samples.shape[0]
        n_obj = samples.shape[1]
        rest = samples.shape[2:]
        flat_samples = samples.view(batch_size * n_obj, *rest)
        embedded_samples = self.vision_dropout(self.feat_model(flat_samples))
        return embedded_samples.view(batch_size, n_obj, -1)

    def get_prototypes(self, samples, targets):
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

        prototypes = self.prototyper(self.embed_images(samples), targets)

        # Applied per prototype rather than to the concatenation, which is the
        #     same thing: dropout masks each element independently.
        return tuple(self.prototype_dropout(p) for p in prototypes)

    def speak(self, samples, targets):
        """
        Produce a message, the symbol embeddings behind it, and the concepts
            that prompted it, all from a single pass.

        This is what compositionality analysis needs: the soft signal
            distances compare symbol embeddings between symbols that were
            actually emitted, and semantic distance is measured between the
            concepts, so all three must come from the same forward pass.
            Fetching them through separate calls would resample both the
            vision dropout mask and (for the GRU speaker) the message itself.

        Returns:
            messages: (batch, message_length, vocabulary + 4)
            symbol_embeddings: (batch, message_length - 2, embedding size),
                one dense contextual vector per content symbol, positionally
                aligned with the content symbols of `messages`, i.e. with the
                output of `trim_messages`
            concepts: (batch, 2 * referent embedding size)
        """
        prototypes = self.get_prototypes(samples, targets)
        messages, symbol_embeddings = self.language_model.decode(prototypes)
        return messages, symbol_embeddings, torch.cat(prototypes, 1)

    def forward(
        self,
        samples,
        targets,
        **kwargs
    ):
        # Prototype once and reuse: a second `get_prototypes` call would re-run
        #     the vision model under a fresh `vision_dropout` mask, so the
        #     returned concepts would not be the ones that produced the message.
        prototypes = self.get_prototypes(samples, targets)

        messages = self.language_model(
            prototypes,
            **kwargs
        )

        return messages, torch.cat(prototypes, 1)

    def reset_parameters(self):
        if hasattr(self.feat_model, 'reset_parameters'):
            self.feat_model.reset_parameters()
        self.prototyper.reset_parameters()
        self.language_model.reset_parameters()