import json
import sys
import os 

import argparse

def get_caller_callee():
    """
    Get callers and callees of binary functions.
    The output is a json file in the format of:
        {
            func_name: {
                'caller': [], 
                'callee': []
            }
        }.
    
    Assume your input binary path is dataset_folder/arch/opt/project_name/binary_name, the output json file is saved as 
    ./icfg/arch/opt/project_name/binary_name.json, where arch is the architecture, opt is the optimization level, 
    project_name is the open source project name, and binary_name is the binary name.

    For example, the input binary path is ./dataset/x86/O0/openssl/ssl, the output json file is saved as 
    ./icfg/x86/O0/openssl/ssl.json.
    """
    parser = argparse.ArgumentParser(description='parse the interprocedural control flow graph of a binary', prefix_chars='+')
    parser.add_argument('++icfg_dir', type=str, nargs=1,
                    help='where icfg is output', default=['./icfg'])
    args = parser.parse_args(args=getScriptArgs())
    output_dir = args.icfg_dir[0]

    file_path = str(getProgramFile())
    binary_name = os.path.basename(file_path)

    # binary_project_path = file_path.split('/')[-4:-1]
    # binary_project_path = '/'.join(binary_project_path)
    binary_project_path = binary_name
    output_dir = os.path.join(output_dir, binary_project_path)

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    res = {}

    getCurrentProgram().setImageBase(toAddr(0), 0)
    ref = currentProgram.getReferenceManager()
    function = getFirstFunction()

    while function is not None:
        func_name = function.name
        res[func_name] = {
            'callee': [],
            'caller': []
        }

        callees = function.getCalledFunctions(ghidra.util.task.TaskMonitor.DUMMY)
        callers = function.getCallingFunctions(ghidra.util.task.TaskMonitor.DUMMY)

        # for caller in callers:
        #     res[func_name]['caller'].append(caller.name)

        listing = currentProgram.getListing()
        instructions = listing.getInstructions(function.getEntryPoint(), True)
        for instruction in instructions:
            addr = instruction.getAddress()
            mnemonic = instruction.getMnemonicString()
            mnemonic = mnemonic.encode('utf-8').upper()
            
            if mnemonic == 'CALL' or mnemonic == 'BL' or mnemonic == 'BLX' or mnemonic == "JAL": # or mnemonic == "JALR"
                output = str(getCodeUnitFormat().getRepresentationString(currentProgram.getListing().getCodeUnitAt(addr)))                

                if instruction.getRegister(0) or ('[' in output) or ("ptr" in output and "word" in output):
                    continue
                # elif "external" in output.lower():
                #     callee_name = output.strip()
                #     print("external: " + callee_name)
                #     res[func_name]['callee'].append(callee_name)
                else:
                    # print("internal before: " + output.strip())
                    callee_name = output.strip().split()[1]                   

                    # print("internal after: " + callee_name)
                    res[func_name]['callee'].append(callee_name)
                    
                    if callee_name not in res:
                        res[callee_name] = {
                            'callee': [],
                            'caller': []
                        }
                    res[callee_name]['caller'].append(func_name) 
            
            if getFunctionContaining(addr) != function:
                break
        
        function = getFunctionAfter(function)

    with open(os.path.join(output_dir, 'icfg.json'), 'w') as f:
        json.dump(res, f)
        print('[*] The interprocedural CFG be saved in: ' + output_dir)

if __name__ == '__main__':
    get_caller_callee()
    
