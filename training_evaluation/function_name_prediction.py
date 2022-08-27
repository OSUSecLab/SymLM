#!/usr/bin/env python
# encoding: utf-8

import logging
import math
import os
import sys
from fairseq.models.roberta_mf.function_name_prediction import FuncNamePred
import numpy as np
import torch
from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.data import encoders

def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.dataset_impl == 'raw', \
        '--replace-unk requires a raw text dataset (--dataset-impl=raw)'
    if args.results_path is not None:
        dir_path = args.results_path.replace(os.path.basename(args.results_path), '')
        os.makedirs(dir_path, exist_ok=True)
        if os.path.exists(args.results_path):
            os.remove(args.results_path)
    else:
        raise Exception('args.results_path is None!')
    _main(args)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, 'symbols_to_strip_from_output'):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(args):
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=os.environ.get('LOGLEVEL', 'INFO').upper(),
    )
    logger = logging.getLogger('fairseq_cli.generate')

    utils.import_user_module(args)

    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    logger.info('loading model(s) from {}'.format(args.path))

    model = FuncNamePred.from_pretrained(
                model_name_or_path=args.checkpoint_dir,
                checkpoint_file=args.checkpoint_file,
                data_name_or_path=args.data,
            )
    
    if use_cuda:
        model.cuda()

    model.eval()

    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=512,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
        data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)

    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        default_log_format=('tqdm' if not args.no_progress_bar else 'none'),
    )

    dictionary = task.label_dictionary
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if 'net_input' not in sample:
            continue
        
        callee_token_list = []
        caller_token_list = []
        external_list = []

        for i in range(args.num_calls):
            if f'callee_tokens{i+1}' not in sample['net_input']:
                break

            callee_token_list.append(sample['net_input'][f'callee_tokens{i+1}'])
            caller_token_list.append(sample['net_input'][f'caller_tokens{i+1}'])
            external_list.append(sample['net_input'][f'external{i+1}'])

        with torch.no_grad():
            logits, _ = model.model(
                sample['net_input']['src_tokens'],  
                callee_token_list,
                caller_token_list,
                external_list,
                features_only=True,
                classification_head_name='func_name_pred',
            )

            targets = sample['target']
            preds = torch.sigmoid(logits)
            preds_indices = torch.topk(preds, k=10).indices + 3
            preds_values = torch.topk(preds, k=10).values
            preds_string = dictionary.string(preds_indices)
            preds_string = preds_string.split('\n')
            targets_string = dictionary.string(targets).replace(' <pad>', '')
            targets_string = targets_string.split('\n')

            with open(args.evaluation_file, 'a+') as f:
                for i, pred in enumerate(preds_string):
                    print(targets_string[i] + ',' + pred + ',' + np.array2string(preds_values[i].cpu().detach().numpy(), formatter={'float_kind':lambda x: "%.4f" % x}), file=f)

            with open(args.results_path, 'a+') as f:
                for i, pred in enumerate(preds_string):
                    print(pred, file=f)


def add_prediction_args(parser):
    group = parser.add_argument_group("Checkpoints and dataset path")
    group.add_argument('--checkpoint-dir', default='checkpoints/train', type=str,
                       help='Directory of the well-trained model')
    group.add_argument('--checkpoint-file', default='checkpoint_best.pt', type=str,
                       help='Name of the checkpoint file')
    group.add_argument('--evaluation-file', type=str,
                       help='File to save the evaluation inputs')                   

def cli_main():
    parser = options.get_generation_parser()
    add_prediction_args(parser)
    args = options.parse_args_and_arch(parser)
    main(args)

if __name__ == '__main__':
    cli_main()