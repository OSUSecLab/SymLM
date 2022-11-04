#!/usr/bin/env python
# encoding: utf-8

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

@register_criterion('func_name_pred')
class FuncNamePred(FairseqCriterion):

    def __init__(self, task):
        super().__init__(task)
        self.fields = params.fields

    @staticmethod
    def add_args(parser):
        parser.add_argument('--classification-head-name',
                            default='func_name_pred',
                            help='name of the classification head to use')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample."""

        targets = sample['target']
        real_tokens = sample['target'].ne(self.task.label_dictionary.pad())

        callee_token_list = []
        caller_token_list = []
        external_list = []
        for i in range(10):
            if f'callee_tokens{i+1}' not in sample['net_input']:
                break

            callee_token_list.append(sample['net_input'][f'callee_tokens{i+1}'])
            caller_token_list.append(sample['net_input'][f'caller_tokens{i+1}'])
            external_list.append(sample['net_input'][f'external{i+1}'])

        logits, _ = model(
            sample['net_input']['src_tokens'],  # TODO: the params of model::forward() function are not sure here.
            callee_token_list,
            caller_token_list,
            external_list,
            features_only=True,
            classification_head_name='func_name_pred',
        )

        targets = F.one_hot(targets.long(), num_classes=logits.size()[-1] + 3)
        targets = targets.sum(dim=1)
        targets = targets[:, 3:]
        targets = targets.to(torch.int32)

        sample_size = real_tokens.int().sum().float()
        loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='sum')

        preds = F.sigmoid(logits)
        preds = preds > 0.5
        preds = preds.to(torch.int32)

        true_positve = ((targets == 1) * (preds == 1)).sum()
        false_positve = ((targets == 0) * (preds == 1)).sum()
        false_negative = ((targets == 1) * (preds == 0)).sum()

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'true_positive': true_positve,
            'false_positive': false_positve,
            'false_negative': false_negative,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        def calculate_results(true_positive, false_positive, false_negative):
            # avoid dev by 0
            if true_positive + false_positive == 0:
                return 0, 0, 0
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            return precision, recall, f1

        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)

        if len(logging_outputs) > 0 and 'true_positive' in logging_outputs[0]:
            true_positive = sum(log.get('true_positive', 0) for log in logging_outputs)
            false_positive = sum(log.get('false_positive', 0) for log in logging_outputs)
            false_negative = sum(log.get('false_negative', 0) for log in logging_outputs)
            # ntype_pred = sum(log.get('ntype_pred', 0) for log in logging_outputs)

            precision, recall, f1 = calculate_results(true_positive, false_positive, false_negative)
            metrics.log_scalar('precision', 100.0 * precision, sample_size, round=1)
            metrics.log_scalar('recall', 100.0 * recall, sample_size, round=1)
            metrics.log_scalar('F1', 100.0 * f1, sample_size, round=1)
            # metrics.log_scalar('accuracy', 100.0 * ncorrect_total / sample_size, sample_size, round=1)

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