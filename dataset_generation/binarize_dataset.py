from multiprocessing import Pool
import subprocess
from itertools import product
import os
import argparse

fields = ['static', 'inst_pos_emb', 'op_pos_emb', 'arch_emb', 'byte1', 'byte2', 'byte3', 'byte4']
targets = ['self']

def run(data_src_dir, data_bin_dir, target, field):
    subprocess.run([
        'fairseq-preprocess', '--only-source',
        '--srcdict', f'vocabulary/{field}/dict.txt',
        '--trainpref', f'{data_src_dir}/train/{target}/input.{field}',
        '--validpref', f'{data_src_dir}/valid/{target}/input.{field}',
        '--testpref', f'{data_src_dir}/test/{target}/input.{field}',
        '--destdir', f'{data_bin_dir}/{target}/{field}',
        '--workers',
        '40'
    ])

def main():
    parser = argparse.ArgumentParser(description='Output ground truth')
    parser.add_argument('--data_src_dir', type=str, nargs=1,
                    help='directory where the dataset to be binarized is stored')
    parser.add_argument('--data_bin_dir', type=str, nargs=1,
                    help='directory where the binarized result is')
    parser.add_argument('--topK', type=int, nargs=1, default=[2],
                    help='number of top popular callers (callees) to be selected')
    args = parser.parse_args()
    data_src_dir = args.data_src_dir[0]
    data_bin_dir = args.data_bin_dir[0]
    topK = args.topK[0]

    if data_src_dir[-1] == '/':
        data_src_dir = data_src_dir[:-1]
    if data_bin_dir[-1] == '/':
        data_bin_dir = data_bin_dir[:-1]

    for i in range(topK):
        targets.append(f"caller{i+1}")
        targets.append(f"internal_callee{i+1}")

    # binarize fields
    with Pool() as pool:
        pool.starmap(run, product([data_src_dir], [data_bin_dir], targets, fields))

    # binarize labels
    for target in ['self'] + [f'external_callee{i+1}' for i in range(topK)]:
        if target == 'self':
            src_dict = 'vocabulary/label/dict.txt'
        else:
            src_dict = 'vocabulary/external_label/dict.txt'
        subprocess.run([
            'fairseq-preprocess', '--only-source',
            '--srcdict', src_dict,
            '--trainpref', f'{data_src_dir}/train/{target}/input.label',
            '--validpref', f'{data_src_dir}/valid/{target}/input.label',
            '--testpref', f'{data_src_dir}/test/{target}/input.label',
            '--destdir', f'{data_bin_dir}/{target}/label',
            '--workers',
            '40'
        ])

    subprocess.run(['cp', '-r', f'vocabulary/cover', f'{data_bin_dir}/self/' ])
    
    print("[*] Binarized dataset under {}".format(data_bin_dir))

if __name__ == '__main__':
    main()
