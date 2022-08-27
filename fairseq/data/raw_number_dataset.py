#!/usr/bin/env python
# encoding: utf-8
"""
@author: Xin Jin
@license: (C) Copyright 2013-2019.
@contact: xin.jin0010@gmail.com
@software: pycharm
@file: raw_number_dataset.py
@time: 3/10/22 8:27 PM
@desc:
"""
import torch

from . import FairseqDataset

class RawNumberDataset(FairseqDataset):

    def __init__(self, labels):
        super().__init__()
        if isinstance(labels[0], list):
            self.labels = labels
        else:
            self.labels = torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, index):
        return torch.tensor(self.labels[index], dtype=torch.int64)

    def __len__(self):
        return len(self.labels)

    def collater(self, samples):
        if torch.is_tensor(samples[0]):
            return torch.stack(samples, dim=0)
        return torch.tensor(samples, dtype=torch.int64)