import os
import shutil
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Output ground truth')
    parser.add_argument('--src_file', type=str, nargs=1,
                    help='file where function names are')
    parser.add_argument('--output_dir', type=str, nargs=1,
                    help='directory where the generated vocabulary will be stored')

    args = parser.parse_args()
    src_file = args.src_file[0]
    output_dir = args.output_dir[0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    subprocess.run([
        'fairseq-preprocess', '--only-source',
        '--trainpref', src_file,
        '--destdir', output_dir,
        '--workers', '40'
    ])

    print("[*] Generated vocabulary at {}".format(os.path.join(output_dir, 'dict.txt')))

if __name__ == '__main__':
    main()