import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--internal_vocabulary_path', type=str, 
                        help='Path to vocabulary file of internal function name words.')
    parser.add_argument('--external_vocabulary_path', type=str, 
                        help='Path to vocabulary file of external function names.')
    args = parser.parse_args()

    with open(args.internal_vocabulary_path, 'r') as f:
        # +1 is to add the <UNK> token for OOV word prediction.
        print(f"NUM_CLASSES={len(f.readlines())+1}")

    with open(args.external_vocabulary_path, 'r') as f:
        print(f"NUM_EXTERNAL={len(f.readlines())}")


if __name__ == '__main__':
    main()