#!/usr/bin/env python
# encoding: utf-8
"""
@author: Xin Jin
@license: (C) Copyright 2013-2019.
@contact: xin.jin0010@gmail.com
@software: pycharm
@file: function_name_translation.py
@time: 12/28/21 3:27 PM
@desc:
"""
import logging

from .nested_dictionary_dataset import NestedDictionaryDataset
from .nested_dictionary_dataset import _unflatten
from . import data_utils

logger = logging.getLogger(__name__)

class FuncNameDataset(NestedDictionaryDataset):

    def __init__(self, defn, tgt_dict, sizes=None, input_feeding=True, left_pad=False):
        super().__init__(
            defn=defn,
            sizes=sizes
        )
        self.tgt_dict = tgt_dict
        self.input_feeding = input_feeding
        self.left_pad = left_pad

    def collater(self, samples):
        """
        add prev_output_tokens to samples['net_input']
        """
        if len(samples) == 0:
            return {}
        pad_idx = self.tgt_dict.pad()
        eos_idx = self.tgt_dict.eos()
        def merge(key, left_pad, move_eos_to_beginning=False):
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                pad_idx, eos_idx, left_pad, move_eos_to_beginning,
            )

        # logging.info(f"size of samples: {len(samples)}")
        if len(samples) > 0:
            assert 'target' in samples[0], "the target key is expected in samples"
        else:
            # logging.info(samples)
            samples = super().collater(samples=samples)
            # print(samples)
            # logging.info("processed samples: " + str(samples))
            # print(samples)
            return samples

        prev_output_tokens = None
        if samples[0].get('prev_output_tokens', None) is not None:
            prev_output_tokens = merge('prev_output_tokens', left_pad=self.left_pad)
        elif self.input_feeding:
            prev_output_tokens = merge(
                'target',
                left_pad=self.left_pad,
                move_eos_to_beginning=True,
            )
        samples = super().collater(samples=samples)

        if prev_output_tokens is not None:
            samples['net_input']['prev_output_tokens'] = prev_output_tokens
        return samples