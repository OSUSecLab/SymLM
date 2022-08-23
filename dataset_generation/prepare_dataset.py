import glob
import json
import os
import random
import re
import sys
import argparse

from capstone import *
from elftools.elf.elffile import ELFFile
import traceback
from collections import Counter
import shutil

class params:
    fields = ['static', 'inst_pos_emb', 'op_pos_emb', 'arch_emb', 'byte1', 'byte2', 'byte3', 'byte4', 'label']
    context_fields = fields[:-1]
    dummy_sequence = {
        'x64': {'static': ',', 'inst_pos_emb': '0', 'op_pos_emb': '0', 'arch_emb': 'x64', 'byte1': '##', 'byte2': '##', 'byte3': '##', 'byte4': '##'},
        'x86': {'static': ',', 'inst_pos_emb': '0', 'op_pos_emb': '0', 'arch_emb': 'x86', 'byte1': '##', 'byte2': '##', 'byte3': '##', 'byte4': '##'},
        'arm': {'static': ',', 'inst_pos_emb': '0', 'op_pos_emb': '0', 'arch_emb': 'arm', 'byte1': '##', 'byte2': '##', 'byte3': '##', 'byte4': '##'},
        'mips': {'static': ',', 'inst_pos_emb': '0', 'op_pos_emb': '0', 'arch_emb': 'mips', 'byte1': '##', 'byte2': '##', 'byte3': '##', 'byte4': '##'},
    }

def get_function_reps(die):
    functions = []
    for child_die in die.iter_children():

        if child_die.tag.split('_')[-1] == 'subprogram':
            function = {}
            try:
                function['start_addr'] = child_die.attributes['DW_AT_low_pc'][2]
                function['end_addr'] = function['start_addr'] + child_die.attributes['DW_AT_high_pc'][2]
                function['name'] = child_die.attributes['DW_AT_name'][2].decode('utf-8')
                functions.append(function)
            except KeyError as e:
                print(traceback.format_exc())
                continue

    return functions

def tokenize(s):
    s = s.replace(',', ' , ')
    s = s.replace('[', ' [ ')
    s = s.replace(']', ' ] ')
    s = s.replace(':', ' : ')
    s = s.replace('*', ' * ')
    s = s.replace('(', ' ( ')
    s = s.replace(')', ' ) ')
    s = s.replace('{', ' { ')
    s = s.replace('}', ' } ')
    s = s.replace('#', '')
    s = s.replace('$', '')
    s = s.replace('!', ' ! ')

    s = re.sub(r'-(0[xX][0-9a-fA-F]+)', r'- \1', s)
    s = re.sub(r'-([0-9a-fA-F]+)', r'- \1', s)

    return s.split()

def byte2seq(value_list):
    return [value_list[i:i + 2] for i in range(len(value_list) - 2)]

def rank_elements(target_list):
    """
    rank the elements in target_list, return the unique elements in the order of their ranks
    """
    counts = Counter(target_list)
    res = counts.most_common()
    return [x[0] for x in res]

def rank_calling_context(calling_context_dict):
    """
    For each function, rank callee and caller functions based on their frequency
    """
    res = {}
    for func_name, calling_context in calling_context_dict.items():
        callers = calling_context['caller']
        callees = calling_context['callee']
        if len(callers) >= 2:
            callers = rank_elements(callers)
        # else:
        #     callers = callers + ["##"] * (2 - len(callers))
        if len(callees) >= 2:
            callees = rank_elements(callees)
        # else:
        #     callees = callees + ["##"] * (2 - len(callees))

        res[func_name] = {'caller': callers, 'callee': callees}
    return res

def hex2str(s, b_len=8):
    num = s.replace('0x', '')

    # handle 64-bit cases, we choose the lower 4 bytes, thus 8 numbers
    if len(num) > b_len:
        num = num[-b_len:]

    num = '0' * (b_len - len(num)) + num
    return num

def get_num_lines(file_name):
    with open(file_name) as f:
        return sum(1 for _ in f)

def main():
    parser = argparse.ArgumentParser(description='Output ground truth')
    parser.add_argument('--output_dir', type=str, nargs=1,
                    help='directory where ground truth is output')
    parser.add_argument('--input_binary_path', type=str, nargs=1,
                    help='directory where the input binary is')
    parser.add_argument('--icfg_dir', type=str, nargs=1,
                    help='directory where the icfg file is (same as the icfg folder for get_calling_context.py',
                    default=['./icfg/'])
    parser.add_argument('--arch', type=str, nargs=1,
                    help='architecture of binary, currently support x86, x64, mips and arm')
    parser.add_argument('--topK', type=int, nargs=1, default=[2],
                    help='number of top popular callers (callees) to be selected')

    args = parser.parse_args()
    output_dir = args.output_dir[0]
    file_path = args.input_binary_path[0]
    icfg_dir = args.icfg_dir[0]
    topK = args.topK[0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # file_list = glob.glob(os.path.join(input_dir, '*'), recursive=True)

    # for file_path in file_list:
    # binary_project_path = file_path.split('/')[-4:-1]
    # binary_project_path = '/'.join(binary_project_path)
    binary_name = os.path.basename(file_path)        
    metadata_dir = os.path.join(icfg_dir, binary_name)
    # binary_name = os.path.basename(file_path)

    # check if the calling context metdata file exists
    metadata_file_path = os.path.join(metadata_dir, 'icfg.json')
    if not os.path.exists(metadata_file_path):
        print('[-]', f"icfg file {metadata_file_path}: not exists, exit")
        return
    
    # create output folder for an individual binary
    binary_output_dir = os.path.join(output_dir, binary_name)
    if not os.path.exists(binary_output_dir):
        print('[*]', f"create output folder for an individual binary: {binary_output_dir}")
        os.makedirs(binary_output_dir)
    for i in range(topK):
        if not os.path.exists(os.path.join(binary_output_dir, 'self')):
            os.makedirs(os.path.join(binary_output_dir, 'self'))
        if not os.path.exists(os.path.join(binary_output_dir, f'caller{i+1}')):
            os.makedirs(os.path.join(binary_output_dir, f'caller{i+1}'))
        if not os.path.exists(os.path.join(binary_output_dir, f'internal_callee{i+1}')):
            os.makedirs(os.path.join(binary_output_dir, f'internal_callee{i+1}'))
        if not os.path.exists(os.path.join(binary_output_dir, f'external_callee{i+1}')):
            os.makedirs(os.path.join(binary_output_dir, f'external_callee{i+1}'))
    
    # load calling context metadata file
    with open(metadata_file_path, 'r') as f:
        print('[*]', "load calling context metadata file: " + metadata_file_path)
        calling_context_dict = json.load(f)

    # select top popular callers and callees
    target_context_dict = rank_calling_context(calling_context_dict)

    func_file = {field: open(os.path.join(binary_output_dir, 'self', f'input.{field}'), 'w') for field in params.fields}
    caller_files = []
    for i in range(topK):
        caller_file = {field: open(os.path.join(binary_output_dir, f'caller{i+1}', f'input.{field}'), 'w') for field in params.context_fields}
        caller_files.append(caller_file)

    internal_callee_files = []
    for i in range(topK):
        internal_callee_file = {field: open(os.path.join(binary_output_dir, f'internal_callee{i+1}', f'input.{field}'), 'w') for field in params.context_fields}
        internal_callee_files.append(internal_callee_file)
    
    external_callee_files = []
    for i in range(topK):
        external_callee_file = open(os.path.join(binary_output_dir, f'external_callee{i+1}', f'input.label'), 'w')
        external_callee_files.append(external_callee_file)

    func_sequence_dict = {}
    target_funcs = []

    with open(file_path, 'rb') as f:
        elffile = ELFFile(f)
        dwarf = elffile.get_dwarf_info()

        # disassemble the byte code with capstone
        code = elffile.get_section_by_name('.text')
        opcodes = code.data()
        addr = code['sh_addr']

        if args.arch[0] == "arm":
            md = Cs(CS_ARCH_ARM, CS_MODE_ARM)
        elif args.arch[0] == "x64":
            md = Cs(CS_ARCH_X86, CS_MODE_64)
        elif args.arch[0] == "x86":
            md = Cs(CS_ARCH_X86, CS_MODE_32)
        elif args.arch[0] == "mips":
            md = Cs(CS_ARCH_MIPS, CS_MODE_MIPS32 + CS_MODE_BIG_ENDIAN)

        counter = 0
        for CU in dwarf.iter_CUs():
            function_reps = get_function_reps(CU.get_top_DIE())
            for func in function_reps:
                start_addr = func['start_addr']
                end_addr = func['end_addr']

                func_opcodes = opcodes[start_addr-addr:]

                # input
                static = []
                inst_pos = []
                op_pos = []
                arch = []
                byte1 = []
                byte2 = []
                byte3 = []
                byte4 = []

                # output
                labels = []

                inst_pos_counter = 0

                if func['name'] not in calling_context_dict:
                    continue
                
                try:
                    for address, size, op_code, op_str in md.disasm_lite(func_opcodes, start_addr):
                        if address >= end_addr:
                            break
                        if start_addr <= address and address < end_addr:
                            tokens = tokenize(f'{op_code} {op_str}')

                            for i, token in enumerate(tokens):
                                if '0x' in token.lower():
                                    static.append('hexvar')
                                    bytes = byte2seq(hex2str(token.lower()))
                                    byte1.append(bytes[0])
                                    byte2.append(bytes[1])
                                    byte3.append(bytes[2])
                                    byte4.append(bytes[3])

                                elif token.lower().isdigit():
                                    static.append('num')
                                    bytes = byte2seq(hex2str(hex(int(token.lower()))))
                                    byte1.append(bytes[0])
                                    byte2.append(bytes[1])
                                    byte3.append(bytes[2])
                                    byte4.append(bytes[3])

                                else:
                                    static.append(token)
                                    byte1.append('##')
                                    byte2.append('##')
                                    byte3.append('##')
                                    byte4.append('##')

                                inst_pos.append(str(inst_pos_counter))
                                op_pos.append(str(i))
                                arch.append(args.arch[0])

                            inst_pos_counter += 1

                except CsError as e:
                    print("ERROR: %s" % e)

                # skip functions with too many tokens or too few tokens
                if len(inst_pos) > 510 or len(inst_pos) < 5:
                    continue  
                
                target_funcs.append(func['name'])
                func_sequence_dict[func['name']] = {
                    'static': ' '.join(static), 
                    'inst_pos_emb': ' '.join(inst_pos), 
                    'op_pos_emb': ' '.join(op_pos), 
                    'arch_emb': ' '.join(arch), 
                    'byte1': ' '.join(byte1), 
                    'byte2': ' '.join(byte2), 
                    'byte3': ' '.join(byte3), 
                    'byte4': ' '.join(byte4), 
                    'label': func['name']
                }

            for func_name in target_funcs:

                # step 1: write function instruction sequence to file
                output_sequences = func_sequence_dict[func_name]
                for field in params.fields:
                    func_file[field].write(output_sequences[field] + '\n')

                callers = target_context_dict[func_name]['caller']
                callees = target_context_dict[func_name]['callee']

                # TODO: write sequences of function itself, and its callers and callees into file
                # - for callee, differentate external and internal calls
                # - FIXED: we split in binary level, see previous preprcessing step for train, test, and validation split

                # step 2: get caller sequences and write them into file
                # collect the most frequent caller
                useful_caller_count = 0
                caller_output_sequences = []
                for caller in callers:
                    if caller in func_sequence_dict:
                        useful_caller_count += 1
                        caller_output_sequences.append(func_sequence_dict[caller])
                    if useful_caller_count >= topK:
                        break
                        
                # if there is no enough useful caller, then use dummy sequences which benifits following preprocessing steps
                while useful_caller_count < topK:
                    useful_caller_count += 1
                    caller_output_sequences.append(params.dummy_sequence[args.arch[0]])
                
                # write caller sequences into files
                for i, output_sequence in enumerate(caller_output_sequences):
                    for field in params.context_fields:
                        caller_files[i][field].write(output_sequence[field] + '\n')

                # step 3: get callee sequences and write them into file
                # collect the most frequent callee (for both internal and external callees)
                useful_internal_callee_count = 0
                useful_external_callee_count = 0
                callee_output_sequences = []
                callee_external_labels = []
                for callee in callees:
                    if callee in func_sequence_dict:
                        if useful_internal_callee_count < topK:
                            useful_internal_callee_count += 1
                            callee_output_sequences.append(func_sequence_dict[callee])
                    elif "EXTERNAL" in callee and "::" in callee:
                        if useful_external_callee_count < topK:
                            try:
                                external_callee_name = callee.split("::")[1]
                            except:
                                external_callee_name = "##" # dummy external callee name that benifits following preprocessing steps

                            useful_external_callee_count += 1
                            callee_external_labels.append(external_callee_name)
                
                # if there is no enough useful callee, then use dummy ones which benifits following preprocessing steps
                while useful_internal_callee_count < topK: 
                    useful_internal_callee_count += 1
                    callee_output_sequences.append(params.dummy_sequence[args.arch[0]])
                
                while useful_external_callee_count < topK:
                    useful_external_callee_count += 1
                    callee_external_labels.append("##")

                # write callee sequences into files
                for i, output_sequence in enumerate(callee_output_sequences):  
                    for field in params.context_fields:
                        internal_callee_files[i][field].write(output_sequence[field] + '\n')
                
                for i, label in enumerate(callee_external_labels):
                    external_callee_files[i].write(label + '\n')
                        
    # close all files
    for field in params.fields:
        func_file[field].close()  
    for i in range(topK):
        for field in params.context_fields:
            caller_files[i][field].close()
            internal_callee_files[i][field].close()
        external_callee_files[i].close()  

    # assert that all files have the same number of lines
    num_lines = get_num_lines(os.path.join(binary_output_dir, 'self', f'input.label'))
    # print(num_lines)
    dirs = glob.glob(os.path.join(binary_output_dir, '*'))
    for dir in dirs:
        files = glob.glob(os.path.join(dir, '*'))
        for file in files:
            current_num_lines = get_num_lines(file)
            assert current_num_lines == num_lines, f"number of lines in files are not the same: \n\t {file}: {current_num_lines} \n\t {os.path.join(binary_output_dir, 'self', f'input.label')}: {num_lines}" 

    # remove intermediate icfg file
    if os.path.exists(metadata_dir):
        shutil.rmtree(metadata_dir)
    
    print('[*]', f'Dataset for {file_path} is generated in: {binary_output_dir}')

if __name__ == '__main__':
    main()

    # TODO:
    # - double check if the generated result is correct
    #   - using ghidra to check each of the functions and its callers and callees are correct
                    





