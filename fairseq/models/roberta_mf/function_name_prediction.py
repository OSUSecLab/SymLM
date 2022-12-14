#!/usr/bin/env python
# encoding: utf-8

import logging
import os.path
import sys

import fairseq.checkpoint_utils
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
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
from fairseq.checkpoint_utils import load_checkpoint_to_cpu, load_pretrained_component_from_model

try:
    from command import params
except ImportError:
    from . import params
from collections import OrderedDict

logger = logging.getLogger(__name__)

@register_model('func_name_pred')
class FuncNamePred(FairseqEncoderModel):

    @classmethod
    def hub_models(cls):
        return {
            'roberta.base': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz',
            'roberta.large': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz',
            'roberta.large.mnli': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz',
            'roberta.large.wsc': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.wsc.tar.gz',
        }

    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args
        self.apply(init_bert_params)
        self.classification_heads = nn.ModuleDict()

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

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample

        # get pre-trained model
        encoder = RobertaEncoderMF(args, task.source_dictionary, task.target_cf_dictionary)
        return cls(args, encoder)

    def forward(self, src_tokens, callee_token_list, caller_token_list, external_list,
                features_only=False, return_all_hiddens=False, classification_head_name=None,
                **kwargs):
        if classification_head_name is not None:
            features_only = True

        self_x, self_extra = self.encoder(src_tokens, features_only, return_all_hiddens, **kwargs)
        x = [self_x]

        for callee_tokens in callee_token_list:
            callee_x, _ = self.encoder(callee_tokens, features_only, return_all_hiddens, **kwargs)
            x.append(callee_x)

        for caller_tokens in caller_token_list:
            caller_x, _ = self.encoder(caller_tokens, features_only, return_all_hiddens, **kwargs)
            x.append(caller_x)

        x = torch.concat(x, dim=1)
        x = self.classification_heads['func_name_pred'](x, external_list)
        return x, self_extra

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def register_classification_list(self, name, num_classes=None, num_external=None, num_calls=None, external_emb='one_hot', inner_dim=None, **kwargs):
        """Register a classification list."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )

        self.classification_heads['func_name_pred'] = FuncNameDecoder(
            input_dim=self.args.encoder_embed_dim,
            activation_fn=self.args.pooler_activation_fn,
            num_classes=num_classes,
            num_external=num_external,
            num_calls=num_calls,
            external_emb=external_emb,
            pooler_dropout=self.args.pooler_dropout,
            q_noise=self.args.quant_noise_pq,
            qn_block_size=self.args.quant_noise_pq_block_size,
        )

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = RobertaClassificationHead(
            self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
            self.args.quant_noise_pq,
            self.args.quant_noise_pq_block_size,
        )

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

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''

        # rename decoder -> encoder before upgrading children modules
        for k in list(state_dict.keys()):
            if k.startswith(prefix + 'decoder'):
                new_k = prefix + 'encoder' + k[len(prefix + 'decoder'):]
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = (
            [] if not hasattr(self, 'classification_heads')
            else self.classification_heads.keys()
        )
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            if prefix + 'classification_heads.' + head_name + '.out_proj.weight' in state_dict:
                num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
            else:
                num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.last_proj.weight'].size(0)
            inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)

            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                        num_classes != self.classification_heads[head_name].out_proj.out_features
                        or inner_dim != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    logger.info('Overwriting ' + prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout, q_noise=0, qn_block_size=8):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(
            nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
        )

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class FuncNameDecoder(nn.Module):
    """Head for predicting embeddings for function name."""

    def __init__(
            self,
            input_dim,
            activation_fn,
            num_classes,
            num_external,
            num_calls,
            external_emb,
            pooler_dropout,
            q_noise=0,
            qn_block_size=8,
            topK=3, # parameter for our specific pooling scheme
    ):
        super().__init__()
        self.topK = topK
        self.external_emb = external_emb
        self.num_external = num_external
        if self.external_emb == 'one_hot':
            self.dense = nn.Linear(input_dim * (self.topK + 2) + num_external, input_dim * (self.topK + 2))
        else:
            external_emb_dim = 256
            self.embedding = nn.Embedding(num_embeddings=num_external, embedding_dim=external_emb_dim, padding_idx=1)
            self.dense = nn.Linear(input_dim * (self.topK + 2) + external_emb_dim * 3 * num_calls, input_dim * (self.topK + 2))
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.activation_fn = utils.get_activation_fn(activation_fn)

        self.last_proj = apply_quant_noise_(
            nn.Linear(input_dim * (self.topK + 2), num_classes), q_noise, qn_block_size
        )

    def select_features(self, features):
        x_mean = torch.mean(features, dim=1)
        x_cls = features[:, 0, :]
        x_topk = torch.topk(features, k=self.topK, dim=1).values
        x_topk_flatten = torch.flatten(x_topk, start_dim=1)
        x = torch.concat((x_mean, x_cls, x_topk_flatten), dim=1)
        return x

    def expand(self, x):
        x_expand = x.repeat(1, self.topK)
        batch_size = x.size()[0]
        emb_dim = x.size()[1]
        x_expand = torch.reshape(x_expand, shape=(batch_size, self.topK, emb_dim))
        return x_expand


    def create_heads(self, features):
        x_mean = torch.mean(features, dim=1)
        x_cls = features[:, 0, :]
        x_topk = torch.topk(features, k=self.topK, dim=1).values
        x_mean_expand = self.expand(x=x_mean)
        x_cls_expand = self.expand(x=x_cls)
        x = torch.concat((x_mean_expand, x_cls_expand, x_topk), dim=2)
        return x

    def get_one_hot_embedding(self, tokens):
        emb = F.one_hot(tokens.long(), num_classes= self.num_external + 4)
        emb = emb.sum(dim=1)
        emb = emb[:, 4:]
        emb = emb.to(torch.float32)
        return emb

    def get_embedding(self, tokens):
        embeddings = self.embedding(tokens)
        emb_mean = torch.mean(embeddings, dim=1)
        emb_max = torch.max(embeddings, dim=1).values
        emb_min = torch.min(embeddings, dim=1).values
        emb = torch.concat((emb_mean, emb_max, emb_min), dim=1)
        return emb

    def forward(self, features, external_list, **kwargs):
        if self.external_emb == 'one_hot':
            external_emb = None
            for external_tokens in external_list:
                if external_emb is None:
                    external_emb = self.get_one_hot_embedding(external_tokens)
                else:
                    external_emb = torch.add(external_emb, self.get_one_hot_embedding(external_tokens))
            x = torch.concat([self.select_features(features), external_emb], dim=1)
        else:
            x = [self.select_features(features)]
            for external_tokens in external_list:
                external_emb = self.get_embedding(external_tokens)
                x.append(external_emb)
            x = torch.concat(x, dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.last_proj(x)
        return x

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

        self.lm_head_byte_value_all = RobertaLMHeadRegAll(
            embed_dim=args.encoder_embed_dim,
            activation_fn=args.activation_fn
        )

        self.lm_head_cf = RobertaLMHeadCls(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary_cf),
            activation_fn=args.activation_fn,
        )

    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, masked_tokens=None,
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
        return x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, **unused):
        inner_states, _ = self.sentence_encoder(
            src_tokens,
            last_state_only=not return_all_hiddens,
        )
        features = inner_states[-1].transpose(0, 1)  # T x B x C -> B x T x C
        return features, {'inner_states': inner_states if return_all_hiddens else None}

    def output_layer(self, features, masked_tokens=None, real_cf_tokens=None, **unused):

        return self.lm_head_byte_value_all(features, masked_tokens), self.lm_head_cf(features, real_cf_tokens)

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

@register_model_architecture('func_name_pred', 'func_name_pred')
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