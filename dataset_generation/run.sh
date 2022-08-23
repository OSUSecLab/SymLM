#!/bin/bash

# GHIDRA_ANALYZEHEADLESS_PATH='/home/xin/Documents/project/binary-semantics/parse_stateformer/ghidra_10.1.2_PUBLIC/support/analyzeHeadless'
# GHIDRA_PROJECT_PATH='/home/xin/Documents/project/binary-semantics/parse_stateformer/ghidra_project/'
# GHIDRA_PROJECT_NAME='parser_2/'
# BINARY_PATH='./sample_binary/x64/O0/bc/bc'
# BINARY_ARCHITECTURE='x64'
# DATASET_OUTPUT_DIR='./sample_output/'
GHIDRA_ANALYZEHEADLESS_PATH='' # path to ghidra analyzeHeadless executable
GHIDRA_PROJECT_PATH='' # path to ghidra project
GHIDRA_PROJECT_NAME='' # name of ghidra project
BINARY_PATH=''  # path to binary
BINARY_ARCHITECTURE='' # architecture of binary, options: x86, x64, arm, mips
DATASET_OUTPUT_DIR='' # path to output directory

# generate interprocedural cfg
$GHIDRA_ANALYZEHEADLESS_PATH $GHIDRA_PROJECT_PATH $GHIDRA_PROJECT_NAME -import $BINARY_PATH -readOnly -postScript ./get_calling_context.py

# generate dataset
python3.6 ./prepare_dataset.py \
    --output_dir $DATASET_OUTPUT_DIR \
    --input_binary_path $BINARY_PATH \
    --arch $BINARY_ARCHITECTURE
