# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
try:
    from command import params
except ImportError:
    from . import params


@register_criterion('signature')
class SignatureCriterion(FairseqCriterion):

    def __init__(self, task, classification_head_name):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.fields = params.fields

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--classification-head-name',
                            default='signature_head',
                            help='name of the classification head to use')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
                hasattr(model, 'classification_heads')
                and self.classification_head_name in model.classification_heads
        ), 'model must provide sentence classification head for --criterion=signature'

        logits = model(
            **sample['net_input'],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )[0]
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.int().sum().float()

        lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        loss = F.nll_loss(lprobs, targets, reduction='sum')

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
        }

        preds = logits.argmax(dim=1)
        logging_output['ncorrect_total'] = (preds == targets).sum()
        logging_output['ncorrect'] = ((preds == targets) * (targets != 0)).sum()

        logging_output['ntype'] = (targets != 0).sum().item()
        logging_output['ntype_pred'] = (preds != 0).sum().item()

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)

        if len(logging_outputs) > 0 and 'ncorrect_total' in logging_outputs[0]:
            ncorrect_total = sum(log.get('ncorrect_total', 0) for log in logging_outputs)
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            ntype = sum(log.get('ntype', 0) for log in logging_outputs)
            ntype_pred = sum(log.get('ntype_pred', 0) for log in logging_outputs)

            precision = ncorrect / (ntype_pred + 1e-5)
            recall = ncorrect / (ntype + 1e-5)
            F1 = 2 * (precision * recall) / (precision + recall + 1e-5)
            metrics.log_scalar('precision', 100.0 * precision, sample_size, round=1)
            metrics.log_scalar('recall', 100.0 * recall, sample_size, round=1)
            metrics.log_scalar('F1', 100.0 * F1, sample_size, round=1)
            metrics.log_scalar('accuracy', 100.0 * ncorrect_total / sample_size, sample_size, round=1)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
