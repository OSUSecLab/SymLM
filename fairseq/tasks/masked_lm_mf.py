# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np

from fairseq.data import (
    BytevalueDataset,
    data_utils,
    Dictionary,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    RightPadDataset,
    PrependTokenDataset,
    SortDataset,
    TokenBlockDataset,
    OffsetTokensDataset,
)
from fairseq.tasks import register_task, LegacyFairseqTask
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq import utils
from command import params

logger = logging.getLogger(__name__)


@register_task('masked_lm_mf')
class MaskedLMTaskMF(LegacyFairseqTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--sample-break-mode', default='complete',
                            choices=['none', 'complete', 'complete_doc', 'eos'],
                            help='If omitted or "none", fills each sample with tokens-per-sample '
                                 'tokens. If set to "complete", splits samples only at the end '
                                 'of sentence, but may include multiple sentences per sample. '
                                 '"complete_doc" is similar but respects doc boundaries. '
                                 'If set to "eos", includes only one sentence per sample.')
        parser.add_argument('--tokens-per-sample', default=512, type=int,
                            help='max number of total tokens over all segments '
                                 'per sample for BERT dataset')
        parser.add_argument('--mask-prob', default=0.15, type=float,
                            help='probability of replacing a token with mask')
        parser.add_argument('--leave-unmasked-prob', default=0.1, type=float,
                            help='probability that a masked token is unmasked')
        parser.add_argument('--random-token-prob', default=0.1, type=float,
                            help='probability of replacing a token with a random token')
        parser.add_argument('--freq-weighted-replacement', default=False, action='store_true',
                            help='sample random replacement words based on word frequencies')
        parser.add_argument('--mask-whole-words', default=False, action='store_true',
                            help='mask whole words; you may also want to set --bpe')
        parser.add_argument('--shorten-method', default='none',
                            choices=['none', 'truncate', 'random_crop'],
                            help='if not none, shorten sequences that exceed --tokens-per-sample')
        parser.add_argument('--shorten-data-split-list', default='',
                            help='comma-separated list of dataset splits to apply shortening to, '
                                 'e.g., "train,valid" (default: all dataset splits)')

    def __init__(self, args, dictionary_dict, dictionary_cf):
        super().__init__(args)
        self.dictionary_dict = dictionary_dict
        self.dictionary_cf = dictionary_cf
        self.seed = args.seed

        # add mask token
        self.mask_idx_dict = {}
        for field, dictionary in dictionary_dict.items():
            self.mask_idx_dict[field] = dictionary.add_symbol('<mask>')

        # All field of each token
        self.fields = params.fields

    @classmethod
    def setup_task(cls, args, **kwargs):
        # paths = utils.split_paths(args.data)
        paths = os.listdir(args.data)
        # assert len(paths) > 0
        # assert len(paths) == len(params.fields)
        assert len(paths) == len(params.fields) + 1  # one more for control flow

        dictionary_dict = {}
        for field in params.fields:
            dictionary_dict[field] = Dictionary.load(os.path.join(args.data, field, 'dict.txt'))
            logger.info(f'{field} dictionary: {len(dictionary_dict[field])} types')

        # control flow label
        dictionary_cf = Dictionary.load(os.path.join(args.data, params.field_cf, 'dict.txt'))
        logger.info(f'{params.field_cf} dictionary: {len(dictionary_cf)} types')

        return cls(args, dictionary_dict, dictionary_cf)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        src_tokens = {}
        target = {}
        for field in self.fields:
            split_path = os.path.join(self.args.data, field, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                self.source_dictionary[field],
                self.args.dataset_impl,
                combine=combine,
            )
            if dataset is None:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

            dataset = maybe_shorten_dataset(
                dataset,
                split,
                self.args.shorten_data_split_list,
                self.args.shorten_method,
                self.args.tokens_per_sample,
                self.args.seed,
            )

            # create continuous blocks of tokens
            dataset = TokenBlockDataset(
                dataset,
                dataset.sizes,
                self.args.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary[field].pad(),
                eos=self.source_dictionary[field].eos(),
                break_mode=self.args.sample_break_mode,
            )
            logger.info('field {} loaded {} blocks from: {}'.format(field, len(dataset), split_path))

            # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
            dataset = PrependTokenDataset(dataset, self.source_dictionary[field].bos())

            src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
                dataset,
                self.source_dictionary[field],
                pad_idx=self.source_dictionary[field].pad(),
                mask_idx=self.mask_idx_dict[field],
                seed=self.args.seed,
                mask_prob=self.args.mask_prob if field in self.fields[params.byte_start_pos:] else 0,
                leave_unmasked_prob=self.args.leave_unmasked_prob,
                random_token_prob=self.args.random_token_prob,
                freq_weighted_replacement=self.args.freq_weighted_replacement,
            )

            src_tokens[field] = RightPadDataset(
                src_dataset,
                pad_idx=self.source_dictionary[field].pad()
            )

            # treat byte prediction as discrete tokens
            # target[field] = RightPadDataset(
            #     tgt_dataset,
            #     pad_idx=self.source_dictionary[field].pad()
            # )

            # treat byte as numerical number
            target[field] = BytevalueDataset(
                RightPadDataset(
                    tgt_dataset,
                    pad_idx=self.source_dictionary[field].pad()
                ),
                self.source_dictionary[field]
            )

            # target_value[field] = BytevalueDataset(target[field], self.source_dictionary[field])

        # parse control flow data
        split_path = os.path.join(self.args.data, params.field_cf, split)

        target_cf_dataset = data_utils.load_indexed_dataset(
            split_path,
            self.target_cf_dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        if target_cf_dataset is None:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

        target_cf_dataset = maybe_shorten_dataset(
            target_cf_dataset,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.tokens_per_sample,
            self.args.seed,
        )

        # create continuous blocks of tokens
        target_cf_dataset = TokenBlockDataset(
            target_cf_dataset,
            target_cf_dataset.sizes,
            self.args.tokens_per_sample - 1,  # one less for <s>
            pad=self.target_cf_dictionary.pad(),
            eos=self.target_cf_dictionary.eos(),
            break_mode=self.args.sample_break_mode,
        )
        logger.info('field {} loaded {} blocks from: {}'.format(params.field_cf, len(target_cf_dataset), split_path))

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        target_cf_dataset = PrependTokenDataset(target_cf_dataset, self.target_cf_dictionary.bos())
        target_cf_value = RightPadDataset(
            target_cf_dataset,
            pad_idx=self.target_cf_dictionary.pad()
        )

        net_input = dict()
        net_input['src_tokens'] = src_tokens
        net_input['src_lengths'] = NumelDataset(src_dataset, reduce=False)

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(src_dataset))

        # Net input has multiple fields
        self.datasets[split] = SortDataset(
            NestedDictionaryDataset(
                {
                    'id': IdDataset(),
                    'net_input': net_input,
                    'target': target,
                    'target_cf_value': target_cf_value,
                    'nsentences': NumSamplesDataset(),
                    'ntokens': NumelDataset(src_dataset, reduce=True),
                },
                sizes=[src_dataset.sizes],
            ),
            sort_order=[
                shuffle,
                src_dataset.sizes,
            ],
        )

    @property
    def source_dictionary(self):
        return self.dictionary_dict

    @property
    def target_dictionary(self):
        return self.dictionary_dict

    @property
    def target_cf_dictionary(self):
        return self.dictionary_cf
