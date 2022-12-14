#!/usr/bin/env python
# encoding: utf-8
"""
@author: Xin Jin
@license: (C) Copyright 2013-2019.
@contact: xin.jin0010@gmail.com
@software: pycharm
@file: model_seq2seq.py
@time: 12/27/21 2:43 PM
@desc:
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqDecoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
    TransformerSentenceEncoderMFNAU,
)
from fairseq.modules.transformer_sentence_encoder_mf import init_bert_params
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from .hub_interface import RobertaHubInterface
from fairseq import utils

try:
    from command import params
except ImportError:
    from . import params

logger = logging.getLogger(__name__)

@register_model('func_name_seq2seq_pretrained_lstm')
class FuncNameLSTMTranslationModel(FairseqEncoderDecoderModel):

    @classmethod
    def hub_models(cls):
        return {
            'roberta.base': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz',
            'roberta.large': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz',
            'roberta.large.mnli': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz',
            'roberta.large.wsc': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.wsc.tar.gz',
        }

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)

        self.args = args
        self.apply(init_bert_params)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder-layers', type=int, metavar='L',
                            help='num encoder layers')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='H',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='F',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='A',
                            help='num encoder attention heads')
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--pooler-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use for pooler layer')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN')
        parser.add_argument('--pooler-dropout', type=float, metavar='D',
                            help='dropout probability in the masked_lm pooler layers')
        parser.add_argument('--max-positions', type=int,
                            help='number of positional embeddings to learn')
        parser.add_argument('--load-checkpoint-heads', action='store_true',
                            help='(re-)register and load heads when loading checkpoints')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        parser.add_argument('--untie-weights-roberta', action='store_true',
                            help='Untie weights between embeddings and classifiers in RoBERTa')

        # params for decoder
        parser.add_argument(
            '--decoder-embed-dim', type=int, metavar='N',
            help='dimensionality of the decoder embeddings',
        )
        parser.add_argument(
            '--decoder-hidden-dim', type=int, metavar='N',
            help='dimensionality of the decoder hidden state',
        )
        parser.add_argument(
            '--decoder-dropout', type=float, default=0.1,
            help='decoder dropout probability',
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample

        # get pre-trained model
        pretrained = RobertaEncoderMF(args, task.source_dictionary, task.target_cf_dictionary)

        encoder = SimpleLSTMEncoder(
            args=args,
            dictionary=task.source_dictionary,
            pretrained=pretrained,
        )

        # TODO: confirm task.target_cf_dictionary
        decoder = IncrementalDecoder(
            dictionary=task.label_dictionary,
            encoder_hidden_dim=args.encoder_embed_dim,
            embed_dim=args.decoder_embed_dim,
            hidden_dim=args.decoder_hidden_dim,
            dropout=args.decoder_dropout,
        )

        model = cls(args=args, encoder=encoder, decoder=decoder)

        # Print the model architecture.
        print(model)

        return model

    def forward(
            self, src_tokens, src_lengths, prev_output_tokens,
            features_only=True, return_all_hiddens=False, **kwargs
    ):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return decoder_out

    @property
    def supported_targets(self):
        return {'self'}

    @classmethod
    def from_pretrained(cls, model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.', bpe='gpt2',
                        **kwargs):
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return RobertaHubInterface(x['args'], x['task'], x['models'][0])


    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    # not reachable at the starting point of epoch 1
    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''

        cur_state = self.decoder.state_dict()
        for k, v in cur_state.items():
            k = prefix + 'decoder.' + k
            if k not in state_dict:
                logger.info(f'Adding {k} to state_dict')
                state_dict[k] = v

        cur_state = self.encoder.state_dict()
        for k, v in cur_state.items():
            k_without_pretrained = prefix + k.replace('pretrained.', 'encoder.')
            if k_without_pretrained in state_dict:
                k = 'encoder.' + k
                logger.info(f'{k_without_pretrained} in state_dict, change it to {k}')
                v = state_dict[k_without_pretrained]
                del state_dict[k_without_pretrained]
                state_dict[k] = v
            if k not in state_dict:
                k = 'encoder.' + k
                logger.info(f'Adding {k} to state_dict')
                state_dict[k] = v

        # self.decoder.state_dict()

        # # rename decoder -> encoder before upgrading children modules
        # for k in list(state_dict.keys()):
        #     print(k)
        #     if k.startswith(prefix + 'decoder'):
        #         new_k = prefix + 'encoder' + k[len(prefix + 'decoder'):]
        #         state_dict[new_k] = state_dict[k]
        #         del state_dict[k]

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # # Handle new classification heads present in the state dict.
        # current_head_names = (
        #     [] if not hasattr(self, 'classification_heads')
        #     else self.classification_heads.keys()
        # )
        # keys_to_delete = []
        # for k in state_dict.keys():
        #     if not k.startswith(prefix + 'classification_heads.'):
        #         continue
        #
        #     head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
        #     num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
        #     inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)
        #
        #     if getattr(self.args, 'load_checkpoint_heads', False):
        #         if head_name not in current_head_names:
        #             self.register_classification_head(head_name, num_classes, inner_dim)
        #     else:
        #         if head_name not in current_head_names:
        #             logger.warning(
        #                 'deleting classification head ({}) from checkpoint '
        #                 'not present in current model: {}'.format(head_name, k)
        #             )
        #             keys_to_delete.append(k)
        #         elif (
        #                 num_classes != self.classification_heads[head_name].out_proj.out_features
        #                 or inner_dim != self.classification_heads[head_name].dense.out_features
        #         ):
        #             logger.warning(
        #                 'deleting classification head ({}) from checkpoint '
        #                 'with different dimensions than current model: {}'.format(head_name, k)
        #             )
        #             keys_to_delete.append(k)
        # for k in keys_to_delete:
        #     del state_dict[k]
        #
        # # Copy any newly-added classification heads into the state dict
        # # with their current weights.
        # if hasattr(self, 'classification_heads'):
        #     cur_state = self.classification_heads.state_dict()
        #     for k, v in cur_state.items():
        #         if prefix + 'classification_heads.' + k not in state_dict:
        #             logger.info('Overwriting ' + prefix + 'classification_heads.' + k)
        #             state_dict[prefix + 'classification_heads.' + k] = v


class SimpleLSTMEncoder(FairseqEncoder):

    def __init__(
        self, args, dictionary, pretrained, embed_dim=768, hidden_dim=768, dropout=0.1,
    ):
        super().__init__(dictionary)
        self.args = args

        # Our encoder will embed the inputs before feeding them to the LSTM.
        self.pretrained = pretrained

        self.dropout = nn.Dropout(p=dropout)

        # We'll use a single-layer, unidirectional LSTM for simplicity.
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )

    def forward(self, src_tokens, src_lengths):
        # The inputs to the ``forward()`` function are determined by the
        # Task, and in particular the ``'net_input'`` key in each
        # mini-batch. We discuss Tasks in the next tutorial, but for now just
        # know that *src_tokens* has shape `(batch, src_len)` and *src_lengths*
        # has shape `(batch)`.

        # Note that the source is typically padded on the left. This can be
        # configured by adding the `--left-pad-source "False"` command-line
        # argument, but here we'll make the Encoder handle either kind of
        # padding by converting everything to be right-padded.


        # Embed the source.
        x, _ = self.pretrained(src_tokens)

        # Apply dropout.
        x = self.dropout(x)

        basz, length, = x.size()[0], x.size()[1]
        src_lengths = torch.ones(basz, dtype=torch.int64) * length

        # Pack the sequence into a PackedSequence object to feed to the LSTM.
        x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.cpu(), batch_first=True)

        # Get the output from the LSTM.
        _outputs, (final_hidden, _final_cell) = self.lstm(x)

        # Return the Encoder's output. This can be any object and will be
        # passed directly to the Decoder.
        return {
            # this will have shape `(bsz, hidden_dim)`
            'final_hidden': final_hidden.squeeze(0),
        }

    # Encoders are required to implement this method so that we can rearrange
    # the order of the batch elements during inference (e.g., beam search).
    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to `new_order`.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            `encoder_out` rearranged according to `new_order`
        """
        final_hidden = encoder_out['final_hidden']
        return {
            'final_hidden': final_hidden.index_select(0, new_order),
        }

class IncrementalDecoder(FairseqIncrementalDecoder):
    """
    To learn more about how incremental decoding works, refer to `this blog
    <http://www.telesens.co/2019/04/21/understanding-incremental-decoding-in-fairseq/>`_.
    """

    def __init__(
            self, dictionary, encoder_hidden_dim=768, embed_dim=768, hidden_dim=768,
            dropout=0.1,
    ):
        super().__init__(dictionary)
        self.embed_tokens = nn.Embedding(
            num_embeddings=len(dictionary),
            embedding_dim=embed_dim,
            padding_idx=dictionary.pad(),
        )
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(
            input_size=encoder_hidden_dim + embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=False,
        )

        self.output_projection = nn.Linear(hidden_dim, len(dictionary))

    # We now take an additional kwarg (*incremental_state*) for caching the
    # previous hidden and cell states.
    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        if incremental_state is not None:
            # If the *incremental_state* argument is not ``None`` then we are
            # in incremental inference mode. While *prev_output_tokens* will
            # still contain the entire decoded prefix, we will only use the
            # last step and assume that the rest of the state is cached.
            prev_output_tokens = prev_output_tokens[:, -1:]

        # This remains the same as before.
        bsz, tgt_len = prev_output_tokens.size()

        # TODO: fix encoder_out which mismatches with the real output of encoder
        final_encoder_hidden = encoder_out['final_hidden']

        # get the average of all tokens of encoder ourput
        # final_encoder_hidden = torch.mean(final_encoder_hidden, dim=1)

        x = self.embed_tokens(prev_output_tokens)
        x = self.dropout(x)
        # y = final_encoder_hidden.unsqueeze(1).expand(bsz, tgt_len, -1)
        x = torch.cat(
            [x, final_encoder_hidden.unsqueeze(1).expand(bsz, tgt_len, -1)],
            dim=2,
        )

        # We will now check the cache and load the cached previous hidden and
        # cell states, if they exist, otherwise we will initialize them to
        # zeros (as before). We will use the ``utils.get_incremental_state()``
        # and ``utils.set_incremental_state()`` helpers.
        initial_state = utils.get_incremental_state(
            self, incremental_state, 'prev_state',
        )
        if initial_state is None:
            # first time initialization, same as the original version
            initial_state = (
                final_encoder_hidden.unsqueeze(0),  # hidden
                torch.zeros_like(final_encoder_hidden).unsqueeze(0),  # cell
            )

        # Run one step of our LSTM.
        output, latest_state = self.lstm(x.transpose(0, 1), initial_state)

        # Update the cache with the latest hidden and cell states.
        utils.set_incremental_state(
            self, incremental_state, 'prev_state', latest_state,
        )

        # # T x B x C -> B x T x C
        x = output.transpose(0, 1)

        x = self.output_projection(x)
        return x, None

    # The ``FairseqIncrementalDecoder`` interface also requires implementing a
    # ``reorder_incremental_state()`` method, which is used during beam search
    # to select and reorder the incremental state.
    def reorder_incremental_state(self, incremental_state, new_order):
        # Load the cached state.
        prev_state = utils.get_incremental_state(
            self, incremental_state, 'prev_state',
        )

        # Reorder batches according to *new_order*.
        reordered_state = (
            prev_state[0].index_select(1, new_order),  # hidden
            prev_state[1].index_select(1, new_order),  # cell
        )

        # Update the cached state.
        utils.set_incremental_state(
            self, incremental_state, 'prev_state', reordered_state,
        )


class RobertaEncoderMF(FairseqEncoder):
    """RoBERTa multifield encoder."""

    def __init__(self, args, dictionary, dictionary_cf):
        super().__init__(dictionary)
        self.args = args
        self.fields = params.fields

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))

        self.sentence_encoder = TransformerSentenceEncoderMFNAU(
            padding_idx_dict={field: dictionary[field].pad() for field in dictionary}, # pad index dictionary to determine padding for each field
            vocab_size_dict={field: len(dictionary[field]) for field in dictionary}, # length dictionary for each field
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            layerdrop=args.encoder_layerdrop,
            max_seq_len=args.max_positions,
            encoder_normalize_before=True,
            apply_bert_init=True,
            activation_fn=args.activation_fn,
            q_noise=args.quant_noise_pq,
            qn_block_size=args.quant_noise_pq_block_size,
        )
        args.untie_weights_roberta = getattr(args, 'untie_weights_roberta', False)

        # self.lm_head_cover = RobertaLMHeadCls(
        #     embed_dim=args.encoder_embed_dim,
        #     output_dim=len(dictionary['cover']),
        #     activation_fn=args.activation_fn,
        #     weight=(
        #         self.sentence_encoder.embed_tokens_dict['cover'].weight
        #         if not args.untie_weights_roberta
        #         else None
        #     ),
        # )

        # self.lm_head_byte = nn.ModuleDict(
        #     {field: RobertaLMHeadCls(
        #         embed_dim=args.encoder_embed_dim,
        #         output_dim=len(dictionary[field]),
        #         activation_fn=args.activation_fn,
        #         weight=self.sentence_encoder.byte_emb.weight,
        #     )
        #         for field in self.fields[params.byte_start_pos:]})
        #
        # self.lm_head_byte_value = nn.ModuleDict(
        #     {field: RobertaLMHeadReg(
        #         embed_dim=args.encoder_embed_dim,
        #         activation_fn=args.activation_fn
        #     )
        #         for field in self.fields[params.byte_start_pos:]})
        self.lm_head_byte_value_all = RobertaLMHeadRegAll(
            embed_dim=args.encoder_embed_dim,
            activation_fn=args.activation_fn
        )

        self.lm_head_cf = RobertaLMHeadCls(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary_cf),
            activation_fn=args.activation_fn,
        )

    def forward(self, src_tokens, features_only=True, return_all_hiddens=False, masked_tokens=None,
                real_cf_tokens=None,
                **unused):
        """
        Args:
            src_tokens (LongTensor): dictionary of input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """
        x, extra = self.extract_features(src_tokens, return_all_hiddens=return_all_hiddens)
        if not features_only:
            x = self.output_layer(x, masked_tokens=masked_tokens, real_cf_tokens=real_cf_tokens)
        # return x, extra
        return x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, **unused):
        inner_states, _ = self.sentence_encoder(
            src_tokens,
            last_state_only=not return_all_hiddens,
        )
        features = inner_states[-1].transpose(0, 1)  # T x B x C -> B x T x C
        return features, {'inner_states': inner_states if return_all_hiddens else None}

    def output_layer(self, features, masked_tokens=None, real_cf_tokens=None, **unused):
        # return self.lm_head(features, masked_tokens)

        # only static code, and dynamic bytes are predicted in masked LM
        # field_pred = {'cover': self.lm_head_cover(features, masked_tokens)}

        # field_pred = {}
        # for field in self.fields[params.byte_start_pos:]:
        #     field_pred[field] = self.lm_head_byte[field](features, masked_tokens)
        #     field_pred[f'{field}_value'] = self.lm_head_byte_value[field](features, masked_tokens)
        #
        # return field_pred

        return self.lm_head_byte_value_all(features, masked_tokens), self.lm_head_cf(features, real_cf_tokens)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    # Encoders are required to implement this method so that we can rearrange
    # the order of the batch elements during inference (e.g., beam search).
    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to `new_order`.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            `encoder_out` rearranged according to `new_order`
        """
        final_hidden = encoder_out['final_hidden']
        return {
            'final_hidden': final_hidden.index_select(0, new_order),
        }


class RobertaLMHeadCls(nn.Module):
    """Head for masked language modeling as classification task."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            # print(features.size(), masked_tokens.size())
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class RobertaLMHeadRegAll(nn.Module):
    """Head for masked language modeling as regression task (for all 4 bytes)."""

    def __init__(self, embed_dim, activation_fn):
        super().__init__()
        self.dense = nn.Linear(embed_dim, 2 * embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(2 * embed_dim)
        self.output_dense = nn.Linear(2 * embed_dim, len(params.fields[params.byte_start_pos:]))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.output_dense(x)
        return x


@register_model_architecture('func_name_seq2seq_pretrained_lstm', 'func_name_seq2seq_pretrained_lstm')
def base_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.0)
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)
    args.encoder_layers_to_keep = getattr(args, 'encoder_layers_to_keep', None)
    args.encoder_layerdrop = getattr(args, 'encoder_layerdrop', 0.0)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.decoder_hidden_dim = getattr(args, 'decoder_hidden_dim', 768)
    args.decoder_dropout = getattr(args, 'decoder_dropout', 0.1)
