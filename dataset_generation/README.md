# Dataset Generation

Instructions about how to generate dataset for binaries.

## Setup

* Ghidra installation

For dataset generation, we use Ghidra to parse the binary. Therefore, you need to install Ghidra first (Our scripts have been tested on Ghidra 10.1.2). For more details, please refer to [Ghidra](https://ghidra-sre.org/).

## Usage

* Run Script

The dataset generation script is [`run.sh`](run.sh). Before running it, please set the following variables:

```bash
GHIDRA_ANALYZEHEADLESS_PATH='' # path to ghidra analyzeHeadless executable
GHIDRA_PROJECT_PATH='' # path to ghidra project
GHIDRA_PROJECT_NAME='' # name of ghidra project
BINARY_PATH=''  # path to binary
BINARY_ARCHITECTURE='' # architecture of binary, options: x86, x64, arm, mips
DATASET_OUTPUT_DIR='' # path to output directory
```

Then simply run the script with:

```bash
cd dataset_generation # make sure you are in the dataset_generation folder
bash run.sh
```

* Example

We provide a sample `x64` binary [`bc`](sample_binary/x64/O0/bc/bc). By running our script, the generated dataset is under [`sample_output/bc`](sample_output/bc) and its structure is:

```plaintext
sample_output/bc/
├── caller1 # folder containing sequences of the first caller
│   ├── input.arch_emb
│   ├── input.byte1
│   ├── input.byte2
│   ├── input.byte3
│   ├── input.byte4
│   ├── input.inst_pos_emb
│   ├── input.op_pos_emb
│   └── input.static
├── caller2 
│   ├── input.arch_emb
│   ├── input.byte1
│   ├── input.byte2
│   ├── input.byte3
│   ├── input.byte4
│   ├── input.inst_pos_emb
│   ├── input.op_pos_emb
│   └── input.static
├── external_callee1 # folder containing external callee names of the first external callee
│   └── input.label # external callee names are used for query external function embedding lookup table
├── external_callee2
│   └── input.label
├── internal_callee1 # folder containing sequences of the first internal callee
│   ├── input.arch_emb
│   ├── input.byte1
│   ├── input.byte2
│   ├── input.byte3
│   ├── input.byte4
│   ├── input.inst_pos_emb
│   ├── input.op_pos_emb
│   └── input.static
├── internal_callee2
│   ├── input.arch_emb
│   ├── input.byte1
│   ├── input.byte2
│   ├── input.byte3
│   ├── input.byte4
│   ├── input.inst_pos_emb
│   ├── input.op_pos_emb
│   └── input.static
└── self    # folder containing sequences of function instructions
    ├── input.arch_emb
    ├── input.byte1
    ├── input.byte2
    ├── input.byte3
    ├── input.byte4
    ├── input.inst_pos_emb
    ├── input.label
    ├── input.op_pos_emb
    └── input.static
```
├── caller1 # folder containing sequences of the first caller
│   ├── input.arch_emb
│   ├── input.byte1
│   ├── input.byte2
│   ├── input.byte3
│   ├── input.byte4
│   ├── input.inst_pos_emb
│   ├── input.op_pos_emb
│   └── input.static
├── caller2 
│   ├── input.arch_emb
│   ├── input.byte1
│   ├── input.byte2
│   ├── input.byte3
│   ├── input.byte4
│   ├── input.inst_pos_emb
│   ├── input.op_pos_emb
│   └── input.static
├── external_callee1 # folder containing external callee names of the first external callee
│   └── input.label # external callee names are used for query external function embedding lookup table
├── external_callee2
│   └── input.label
├── internal_callee1 # folder containing sequences of the first internal callee
│   ├── input.arch_emb
│   ├── input.byte1
│   ├── input.byte2
│   ├── input.byte3
│   ├── input.byte4
│   ├── input.inst_pos_emb
│   ├── input.op_pos_emb
│   └── input.static
├── internal_callee2
│   ├── input.arch_emb
│   ├── input.byte1
│   ├── input.byte2
│   ├── input.byte3
│   ├── input.byte4
│   ├── input.inst_pos_emb
│   ├── input.op_pos_emb
│   └── input.static
└── self    # folder containing sequences of function instructions
    ├── input.arch_emb
    ├── input.byte1
    ├── input.byte2
    ├── input.byte3
    ├── input.byte4
    ├── input.inst_pos_emb
    ├── input.label
    ├── input.op_pos_emb
    └── input.static
```