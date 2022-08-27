# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import torch
import torch.nn.functional as F

from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from command import params


@register_criterion('masked_lm_mf')
class MaskedLmLossMF(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, task, tpu=False):
        super().__init__(task)
        self.tpu = tpu

        self.fields = params.fields

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        masked_tokens = sample['target'][self.fields[-1]].ne(self.padding_idx_dict[self.fields[-1]])
        real_cf_tokens = sample['target_cf_value'].ne(self.task.target_cf_dictionary.pad())

        assert torch.all(masked_tokens.eq(sample['target'][self.fields[-3]].ne(self.padding_idx_dict[self.fields[-3]])))

        sample_size = masked_tokens.int().sum()

        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        if self.tpu:
            masked_tokens = None  # always project all tokens on TPU
        elif masked_tokens.device == torch.device('cpu'):
            if not masked_tokens.any():
                masked_tokens = None
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )

        # print(self.task.source_dictionary['static'].string(sample['net_input']['src_tokens']['static']))
        # print(self.task.source_dictionary['byte4'].string(sample['net_input']['src_tokens']['byte4']))
        # exit()

        model_out = model(**sample['net_input'], masked_tokens=masked_tokens, real_cf_tokens=real_cf_tokens)[0]
        logits = model_out[0]
        cf_logits = model_out[1]
        targets = model.get_targets(sample, [logits])
        # targets_values = sample['target_value']

        # which field to predict
        output_langs = params.output_langs
        # for output_lang in output_langs:
        #     assert output_lang in logits.keys()

        targets_stacked = torch.stack([targets[field][masked_tokens] for field in output_langs], dim=1)
        byte_loss = F.mse_loss(
            logits.float(),
            targets_stacked.float(),
            reduction='sum'
        )

        cf_targets = sample['target_cf_value'][real_cf_tokens]
        # print(cf_logits, cf_targets)
        # exit()
        cf_loss = modules.cross_entropy(
            cf_logits.view(-1, cf_logits.size(-1)),
            cf_targets.view(-1),
            reduction='sum',
            ignore_index=self.task.target_cf_dictionary.pad(),
        )
        loss = byte_loss + cf_loss

        if random.random() < 0.001:  # only randomly log some prediction in case screen flushing
            for i, field in enumerate(output_langs):
                print(f'{field} target value:', targets[field][masked_tokens].view(-1)[:6].tolist())
                print(f'{field} pred value:', logits[:6, i].view(-1).tolist())
            print(f'MSE loss: {byte_loss.item()}')

        # loss = 0
        # for field in output_langs:
        #     if masked_tokens is not None:
        #         targets[field] = targets[field][masked_tokens]
        #         targets_values[field] = targets_values[field][masked_tokens]
        #     # print(torch.argmax(logits[field].view(-1, logits[field].size(-1)), dim=1), targets[field].view(-1))
        #     # print(logits[field].view(-1, logits[field].size(-1)).size(), targets[field].view(-1).size())
        #     # exit()
        #
        #     if random.random() < 0.001:  # only randomly log some prediction in case screen flushing
        #         decoded_target = self.task.source_dictionary[field].string(targets[field].view(-1))
        #         decoded_pred = self.task.source_dictionary[field].string(
        #             torch.argmax(logits[field].view(-1, logits[field].size(-1)), dim=1))
        #         print(f'{field} target:', decoded_target.split()[:10])
        #         print(f'{field} pred:', decoded_pred.split()[:10])
        #         print(f'{field} target value:', targets_values[field].view(-1)[:10].tolist())
        #         print(f'{field} pred value:', logits[f'{field}_value'].view(-1)[:10].tolist())
        #
        #     celoss = modules.cross_entropy(
        #         logits[field].view(-1, logits[field].size(-1)),
        #         targets[field].view(-1),
        #         reduction='sum',
        #         ignore_index=self.padding_idx_dict[field],
        #     )
        #     # print(celoss)
        #     loss += params.trace_weights[field] * celoss
        #
        #     # Added MSE loss
        #     mseloss = F.mse_loss(
        #         logits[f'{field}_value'].view(-1),
        #         targets_values[field].view(-1),
        #         reduction='mean'
        #     )
        #     # print(mseloss)
        #     if mseloss.isinf():
        #         print(logits[f'{field}_value'].view(-1)[:10].tolist(), targets_values[field].view(-1)[:10].tolist())
        #     loss += mseloss

        logging_output = {
            'loss': loss if self.tpu else loss.data,
            'byte_loss': loss if self.tpu else byte_loss.data,
            'cf_loss': loss if self.tpu else cf_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        byte_loss_sum = sum(log.get('byte_loss', 0) for log in logging_outputs)
        cf_loss_sum = sum(log.get('cf_loss', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        # for i, log in enumerate(logging_outputs):
        #     print(i, log.get('loss', 0))
        #
        # print(loss_sum, sample_size)
        # exit()
        metrics.log_scalar('loss', loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar('byte_loss', byte_loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar('cf_loss', cf_loss_sum / sample_size, sample_size, round=3)
        # metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
