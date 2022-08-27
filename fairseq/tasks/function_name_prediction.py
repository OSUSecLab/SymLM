#!/usr/bin/env python
# encoding: utf-8

import logging
import os

import numpy as np

from fairseq.data import (
    data_utils,
    Dictionary,
    IdDataset,
    OffsetTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    RightPadDataset,
    PrependTokenDataset,
    SortDataset,
    StripTokenDataset,
    TruncateDataset,
    RawLabelDataset,
    RawNumberDataset,
)
from fairseq.tasks import register_task, LegacyFairseqTask
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq import utils
from command import params

logger = logging.getLogger(__name__)

@register_task('func_name_pred')
class FuncNamePred(LegacyFairseqTask):
    """Task for training function name prediction model"""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='colon separated path to data directories list, \
                                                will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--num-classes', type=int, default=-1,
                            help='size of vocabulary of the internal function name words')
        parser.add_argument('--num-external', type=int, default=-1,
                            help='size of vocabulary of the external function names')
        parser.add_argument('--num-calls', type=int, default=2,
                            help='number of function calls (callers, internal calees and external callees) to be considered')
        parser.add_argument('--external-emb', type=str, default='one_hot',
                            help='external callee embedding method, options: {one_hot, embedding}')
        parser.add_argument('--no-shuffle', action='store_true', default=False)

    def __init__(self, args, data_dictionary_dict, label_dictionary, dictionary_cf, external_dictionary):
        super().__init__(args)
        self.dictionary_dict = data_dictionary_dict
        self.dictionary_cf = dictionary_cf
        self._label_dictionary = label_dictionary
        self.external_dict = external_dictionary
        self._call_dictionary = Dictionary()

        if not hasattr(args, 'max_positions'):
            self._max_positions = 512
        else:
            self._max_positions = args.max_positions

        args.tokens_per_sample = self._max_positions
        self.seed = args.seed
        self.fields = params.fields

    @classmethod
    def load_dictionary(cls, args, filename, source=True, with_mask=True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        if with_mask:
            dictionary.add_symbol('<mask>')
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):

        assert args.num_classes > 0, 'Must set --num-classes'
        assert args.num_external > 0, 'Must set --num-external'
        data_dictionary_dict = {}
        for field in params.fields:
            data_dictionary_dict[field] = cls.load_dictionary(
                args,
                os.path.join(args.data, 'self', field, 'dict.txt'),
                source=True
            )
            logger.info(f'| [input] {field} dictionary: {len(data_dictionary_dict[field])} types')

        # load vocabulary for internal functions
        label_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, 'self', 'label', 'dict.txt'),
            source=False,
            with_mask=False,
        )
        print('| [internal function] dict: {} types'.format(len(label_dict)))

        # control flow label
        dictionary_cf = Dictionary.load(os.path.join(args.data, 'self', params.field_cf, 'dict.txt'))
        logger.info(f'{params.field_cf} dictionary: {len(dictionary_cf)} types')

        # load vocabulary of external callees
        external_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, 'external_callee1', 'label', 'dict.txt'),
            source=False,
            with_mask=False,
        )
        print('| [external function] dictionary: {} types'.format(len(external_dict)))
        print('| [external function] embedding method: {}'.format(args.external_emb))

        return cls(args, data_dictionary_dict, label_dict, dictionary_cf, external_dict)

    def load_dataset_fields(self, split, target, combine=False):
        src_tokens = {}

        for field in self.fields:
            split_path = os.path.join(self.args.data, target, field, split)

            src_dataset = data_utils.load_indexed_dataset(
                split_path,
                self.source_dictionary[field],
                self.args.dataset_impl,
                combine=combine,
            )

            if src_dataset is None:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

            src_tokens[field] = PrependTokenDataset(
                RightPadDataset(
                    StripTokenDataset(
                        TruncateDataset(
                            src_dataset, self.max_positions()),
                        id_to_strip=self.source_dictionary[field].eos()
                    ),
                    pad_idx=self.source_dictionary[field].pad()
                ),
                self.source_dictionary[field].bos()
            )

        return src_tokens, src_dataset

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0

        target = {}

        src_tokens, src_dataset = self.load_dataset_fields(split, target='self')

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(src_dataset))

        net_input = dict()
        net_input['src_tokens'] = src_tokens
        net_input['src_lengths'] = NumelDataset(src_dataset, reduce=False)

        for i in range(self.args.num_calls):

            # load internal callees and callers
            callee_tokens, callee_dataset = self.load_dataset_fields(split, target=f'internal_callee{i+1}')
            caller_tokens, caller_dataset = self.load_dataset_fields(split, target=f'caller{i+1}')
            net_input[f'callee_tokens{i+1}'] = callee_tokens
            net_input[f'callee_lengths{i+1}'] = NumelDataset(callee_dataset, reduce=False)
            net_input[f'caller_tokens{i+1}'] = caller_tokens
            net_input[f'caller_lengths{i+1}'] = NumelDataset(caller_dataset, reduce=False)

            # load external callees
            external_path = os.path.join(self.args.data, f'external_callee{i+1}', 'label', split)
            external_dataset = data_utils.load_indexed_dataset(
                external_path,
                self.external_dict,
                self.args.dataset_impl,
                combine=combine,
            )

            if external_dataset is None:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, external_path))

            net_input[f'external{i+1}'] = RightPadDataset(
                StripTokenDataset(
                    TruncateDataset(
                        external_dataset,
                        self.max_positions(),
                    ), id_to_strip=self.label_dictionary.eos()),
                pad_idx=self.label_dictionary.pad()
            )

        # Net input has multiple fields
        dataset = {
            'id': IdDataset(),
            'net_input': net_input,
            'target': target,
            'nsentences': NumSamplesDataset(),
            'ntokens': NumelDataset(src_dataset, reduce=True),
        }

        label_path = os.path.join(self.args.data, 'self', 'label', split)
        label_dataset = data_utils.load_indexed_dataset(
            label_path,
            self.label_dictionary,
            self.args.dataset_impl,
            combine=combine,
        )

        if label_dataset is None:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, label_path))

        dataset.update(
            target=RightPadDataset(
                # OffsetTokensDataset(
                StripTokenDataset(
                    TruncateDataset(
                        label_dataset,
                        self.max_positions(),
                    ), id_to_strip=self.label_dictionary.eos()),

                pad_idx=self.label_dictionary.pad()
            )
        )

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_dataset.sizes],
        )

        if self.args.no_shuffle:
            self.datasets[split] = nested_dataset
        else:
            # shuffle_sorted = np.sort(shuffle)
            self.datasets[split] = SortDataset(
                nested_dataset,
                # shuffle
                sort_order=[shuffle],
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(self.datasets[split])))
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)

        model.register_classification_list(
            getattr(args, 'classification_head_name', 'func_name_pred'),
            num_classes=self.args.num_classes,
            num_external=self.args.num_external,
            num_calls=self.args.num_calls,
            external_emb=self.args.external_emb,
        )

        return model

    def max_positions(self):
        return self._max_positions

    @property
    def source_dictionary(self):
        return self.dictionary_dict

    @property
    def target_dictionary(self):
        return self.dictionary_dict

    @property
    def label_dictionary(self):
        return self._label_dictionary

    @property
    def target_cf_dictionary(self):
        return self.dictionary_cf

    @property
    def call_dictionary(self):
        return self._call_dictionary