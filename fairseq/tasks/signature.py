# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
    TruncateDataset
)
from fairseq.tasks import register_task, LegacyFairseqTask
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq import utils
from command import params

logger = logging.getLogger(__name__)


@register_task('signature')
class SignatureMF(LegacyFairseqTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--num-classes', type=int, default=2,
                            help='number of classes')
        parser.add_argument('--no-shuffle', action='store_true', default=False)

    def __init__(self, args, data_dictionary_dict, label_dictionary, dictionary_cf):
        super().__init__(args)
        self.dictionary_dict = data_dictionary_dict
        self.dictionary_cf = dictionary_cf
        self._label_dictionary = label_dictionary
        if not hasattr(args, 'max_positions'):
            self._max_positions = (
                args.max_source_positions,
                args.max_target_positions,
            )
        else:
            self._max_positions = args.max_positions
        args.tokens_per_sample = self._max_positions

        self.seed = args.seed

        # add mask token
        # self.mask_idx_dict = {}
        # for field, dictionary in dictionary_dict.items():
        #     self.mask_idx_dict[field] = dictionary.add_symbol('<mask>')

        # All field of each token
        self.fields = params.fields

    @classmethod
    def load_dictionary(cls, args, filename, source=True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol('<mask>')
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        # paths = utils.split_paths(args.data)
        # paths = os.listdir(args.data)
        # assert len(paths) > 0
        # assert len(paths) == len(params.fields)
        assert args.num_classes > 0, 'Must set --num-classes'

        data_dictionary_dict = {}
        for field in params.fields:
            data_dictionary_dict[field] = cls.load_dictionary(
                args,
                os.path.join(args.data, field, 'dict.txt'),
                source=True
            )
            logger.info(f'| [input] {field} dictionary: {len(data_dictionary_dict[field])} types')

        # load label dictionary
        label_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, 'label', 'dict.txt'),
            source=False,
        )
        print('| [label] dictionary: {} types'.format(len(label_dict)))

        # control flow label
        dictionary_cf = Dictionary.load(os.path.join(args.data, params.field_cf, 'dict.txt'))
        logger.info(f'{params.field_cf} dictionary: {len(dictionary_cf)} types')

        return cls(args, data_dictionary_dict, label_dict, dictionary_cf)

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

            src_dataset = data_utils.load_indexed_dataset(
                split_path,
                self.source_dictionary[field],
                self.args.dataset_impl,
                combine=combine,
            )
            if src_dataset is None:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

            src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary[field].bos())
            src_tokens[field] = RightPadDataset(
                TruncateDataset(src_dataset, self.args.max_positions),
                pad_idx=self.source_dictionary[field].pad())

        net_input = dict()
        net_input['src_tokens'] = src_tokens
        net_input['src_lengths'] = NumelDataset(src_dataset, reduce=False)

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(src_dataset))

        # Net input has multiple fields
        dataset = {
            'id': IdDataset(),
            'net_input': net_input,
            'target': target,
            'nsentences': NumSamplesDataset(),
            'ntokens': NumelDataset(src_dataset, reduce=True),
        }

        label_path = os.path.join(self.args.data, 'label', split)
        label_dataset = data_utils.load_indexed_dataset(
            label_path,
            self.label_dictionary,
            self.args.dataset_impl,
            combine=combine,
        )

        if label_dataset is None:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, label_path))

        dataset.update(
            target=OffsetTokensDataset(
                StripTokenDataset(
                    label_dataset,
                    id_to_strip=self.label_dictionary.eos()),
                offset=-self.label_dictionary.nspecial,
            )
        )

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_dataset.sizes],
        )

        if self.args.no_shuffle:
            self.datasets[split] = nested_dataset
        else:
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

        model.register_classification_head(
            getattr(args, 'classification_head_name', 'signature_head'),
            num_classes=self.args.num_classes,
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
