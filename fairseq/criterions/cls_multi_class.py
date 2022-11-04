#!/usr/bin/env python
# encoding: utf-8
"""
@author: Xin Jin
@license: (C) Copyright 2013-2019.
@contact: xin.jin0010@gmail.com
@software: pycharm
@file: cls_multi_class.py
@time: 1/5/22 10:17 PM
@desc:
"""
import math

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
try:
    from command import params
except ImportError:
    from . import params
import logging

logger = logging.getLogger(__name__)

@register_criterion('cls_multi_class')
class CLSMultiClassCriterion(FairseqCriterion):

    def __init__(self, task):
        super().__init__(task)
        # self.classification_head_name = classification_head_name
        self.fields = params.fields

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--classification-head-name',
                            default='cls_multi_class',
                            help='name of the classification head to use')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

                Returns a tuple with three elements:
                1) the loss
                2) the sample size, which is used as the denominator for the gradient
                3) logging outputs to display while training
                """
        # assert (
        #         hasattr(model, 'classification_heads')
        #         and self.classification_head_name in model.classification_heads
        # ), 'model must provide sentence classification head for --criterion=sentence_prediction'

        real_tokens = sample['target'].ne(self.task.label_dictionary.pad())

        sample_size = real_tokens.int().sum().float()

        logits, _ = model(
            **sample['net_input'],  # TODO: the params of model::forward() function are not sure here.
            features_only=True,
            classification_head_name='cls_multi_class',
        )

        targets = model.get_targets(sample, [logits])[real_tokens].view(-1)

        # logits[real_tokens, :], IndexError: too many indices for tensor of dimension 2
        lprobs = F.log_softmax(logits[real_tokens, :], dim=-1, dtype=torch.float32)
        loss = F.nll_loss(lprobs, targets, reduction='sum')

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
        }

        preds = logits[real_tokens, :].argmax(dim=1)
        logging_output['ncorrect_total'] = (preds == targets).sum()
        logging_output['ncorrect'] = ((preds == targets) * (targets != 0)).sum()

        logging_output['ntype'] = (targets != 0).sum().item()
        logging_output['ntype_pred'] = (preds != 0).sum().item()

        return loss, sample_size, logging_output

    def old_forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # real_tokens = sample['target'].ne(self.task.label_dictionary.pad())
        targets = sample['target']

        real_tokens = sample['target'].ne(self.task.label_dictionary.pad())

        # a = self.task.label_dictionary
        # b = a.symbols
        #
        # print(b)

        logits, _ = model(
            **sample['net_input'],  # TODO: the params of model::forward() function are not sure here.
            features_only=True,
            classification_head_name='cls_multi_class',
        )
        # a = targets[real_tokens]
        # a = targets
        targets = F.one_hot(targets.long(), num_classes=logits.size()[-1] + 3)
        # try:
        #     targets = F.one_hot(targets.long(), num_classes=logits.size()[-1] + 3)
        # except:
        #     a = targets
        #     b = a.cpu().detach().numpy()
        #     print(b)
        targets = targets.sum(dim=1)
        targets = targets[:, 3:]
        # b = targets.cpu().detach().numpy()

        # pred = torch.sigmoid(x)
        # loss = F.binary_cross_entropy(pred, y)
        # equals to F.binary_cross_entropy_with_logits(x, y), according to https://zhang-yang.medium.com/how-is-pytorchs-binary-cross-entropy-with-logits-function-related-to-sigmoid-and-d3bd8fb080e7

        # lprobs = F.logsigmoid(logits)
        # sample_size = real_tokens.int().sum().float()

        loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='sum')

        preds = F.sigmoid(logits) > 0.5
        preds = preds.to(torch.float32)

        sample_size = sample['target'].size(0) if 'target' in sample else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True