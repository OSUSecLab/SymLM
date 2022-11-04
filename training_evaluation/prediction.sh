#!/usr/bin/env bash

CHECKPOINT_PATH=checkpoints/train   # Directory to save checkpoints
NUM_CLASSES=3069    # Vocabulary size of internal function name words, plus one for <unk> token (OOV words)
NUM_EXTERNAL=948    # Vocabulary size of external function names
NUM_CALLs=1         # Number of callers/internal callees/external calees per batch (batch size)
MAX_SENTENCEs=8   # Number of sequences per batch (batch size)
EXTERNAL_EMB="embedding"    # External callee embedding methods, options: (one_hot, embedding)
DATASET_PATH="data_bin"      # Path to the binarized dataset
RESULT_FILE="training_evaluation/prediction_evaluation/prediction_result.txt" # File to save the prediction results
EVALUATION_FILE="training_evaluation/prediction_evaluation/evaluation_input.txt" # File to save the evaluation input

CUDA_VISIBLE_DEVICES=1,2 python training_evaluation/function_name_prediction.py \
  $DATASET_PATH \
  --checkpoint-dir $CHECKPOINT_PATH \
  --checkpoint-file checkpoint_best.pt \
  --path $CHECKPOINT_PATH/checkpoint_best.pt \
  --task func_name_pred \
  --criterion func_name_pred \
  --num-external $NUM_EXTERNAL --external-emb $EXTERNAL_EMB \
  --num-calls $NUM_CALLs \
  --num-classes $NUM_CLASSES --max-sentences $MAX_SENTENCEs \
  --gen-subset test \
  --results-path $RESULT_FILE --evaluation-file $EVALUATION_FILE