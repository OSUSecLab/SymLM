#!/usr/bin/env bash

CHECKPOINT_PATH=checkpoints/train   # path to save checkpoints
mkdir -p $CHECKPOINT_PATH
rm -f $CHECKPOINT_PATH/checkpoint_best.pt
cp checkpoints/pretrain/checkpoint_best.pt $CHECKPOINT_PATH/

TOTAL_UPDATES=20000    # Total number of training steps
WARMUP_UPDATES=100    # Warmup the learning rate over this many updates
PEAK_LR=1e-5          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512 # Max sequence length
MAX_POSITIONS=512     # Num. positional embeddings (usually same as above)
MAX_SENTENCES=4       # Number of sequences per batch (batch size)
NUM_CLASSES=3069    # Vocabulary size of internal function name words, plus one for <unk> token (OOV words)
NUM_EXTERNAL=948    # Vocabulary size of external function names
NUM_CALLs=1         # Number of callers/internal callees/external calees per batch (batch size)
ENCODER_EMB_DIM=768 # Embedding dimension for encoder
ENCODER_LAYERS=8    # Number of encoder layers
ENCODER_ATTENTION_HEADS=12  # Number of attention heads for the encoder
TOTAL_EPOCHs=25    # Total number of training epochs
EXTERNAL_EMB="embedding"    # External callee embedding methods, options: (one_hot, embedding)
DATASET_PATH="data_bin"      # Path to the binarized dataset

CUDA_VISIBLE_DEVICES=0 python train.py \
  $DATASET_PATH \
  --ddp-backend=no_c10d \
  --num-classes $NUM_CLASSES --num-external $NUM_EXTERNAL \
  --external-emb $EXTERNAL_EMB --num-calls $NUM_CALLs \
  --task func_name_pred --criterion func_name_pred --arch func_name_pred \
  --reset-optimizer --reset-dataloader --reset-meters \
  --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
  --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES \
  --total-num-update $TOTAL_UPDATES \
  --max-epoch $TOTAL_EPOCHs \
  --update-freq 2 \
  --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
  --best-checkpoint-metric F1 --maximize-best-checkpoint-metric \
  --encoder-layers $ENCODER_LAYERS --encoder-embed-dim $ENCODER_EMB_DIM --encoder-attention-heads $ENCODER_ATTENTION_HEADS \
  --max-positions $MAX_POSITIONS --max-sentences $MAX_SENTENCES \
  --max-update $TOTAL_UPDATES --log-format json --log-interval 10 \
  --no-epoch-checkpoints --save-dir $CHECKPOINT_PATH/ \
  --restore-file $CHECKPOINT_PATH/checkpoint_best.pt 