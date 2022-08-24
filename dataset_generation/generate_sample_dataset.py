from cgi import test
import os.path
from pydoc import splitdoc
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import glob
import threading
import argparse
import time

ghidra_projects = [f'parser_{i}/' for i in range(10)]
lock = threading.Lock()
thread_num = 4
executor = ThreadPoolExecutor(max_workers=thread_num)
analyzeHeadless_path = "/home/xin/Documents/project/binary-semantics/parse_stateformer/ghidra_10.1.2_PUBLIC/support/analyzeHeadless"
ghidra_project_path = "/home/xin/Documents/project/binary-semantics/parse_stateformer/ghidra_project/"
# project_name = "parser_2/"
result_folder = "/home/xin/Documents/project/binary-semantics/parse_stateformer/sample_dataset/"


def get_binary_project():
    projects = []
    with open('/home/xin/Documents/project/binary-semantics/parse_stateformer/lists/x64-O0.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')
            projects.append(line)
    print(f"Load {len(projects)} projects for x64-O0")
    return projects

def process_one_project(binary_project_path, ghidra_project, finished):
    print(f"[*] hold {ghidra_project} for {binary_project_path}")
    binaries = glob.glob(os.path.join(binary_project_path, '*'))
    print('=' * 20, f'with {len(binaries)} binaries, start on {binary_project_path}', '=' * 20)
    project_name = ghidra_project

    for binary in binaries:
        paths = binary.split('/')
        binary_project_name = paths[-2]
        if os.path.join(paths[-2], paths[-1]) in finished:
            print(f"[*] {binary} has been finished")
            continue
        icfg_dir = os.path.join('./icfg', binary_project_name)
        
        cmd = f"{analyzeHeadless_path} {ghidra_project_path} {project_name} -import {binary} -readOnly -postScript ./get_calling_context.py ++icfg_dir {icfg_dir}"
        os.system(cmd)

        DATASET_OUTPUT_DIR = os.path.join(result_folder, binary_project_name)
        cmd = f"python3.6 ./prepare_dataset.py --output_dir {DATASET_OUTPUT_DIR} --input_binary_path {binary} --arch x64 --icfg_dir {icfg_dir}"
        os.system(cmd)
    ghidra_projects.append(ghidra_project)
    print(f"[*] release {ghidra_project} after finishing {binary_project_path}")
    

def main():
    projects = get_binary_project()
    finished = get_finished()
    for project in projects:
        while len(ghidra_projects) == 0:
            print("Wait for ghidra project: 1 sec")
            time.sleep(1)
        ghidra_project = ghidra_projects.pop()
        executor.submit(process_one_project, 
                        binary_project_path=f"/home/xin/Documents/project/binary-semantics/parse_stateformer/{project}",
                        ghidra_project=ghidra_project,
                        finished=finished)
        while executor._work_queue.qsize() > thread_num:
            print("Wait for executor: 1 sec", executor._work_queue.qsize())
            time.sleep(1)

def test_one_project():
    binary_project_path = '/home/xin/Documents/project/binary-semantics/parse_stateformer/stateformer_binaries/x64/O0/bc'
    ghidra_project = 'parser_2/'
    process_one_project(binary_project_path, ghidra_project)

def get_finished():
    finished = set()
    for dir in os.listdir(result_folder): 
        for binary in os.listdir(os.path.join(result_folder, dir)):
            finished.add(os.path.join(dir, binary))
    # print(finished)
    print("[*] Finished on:", len(finished))
    return finished

def copy_lines(src_file, des_file):
    total = 0
    with open(des_file, 'a+') as f_des:
        with open(src_file, 'r') as f_src:
            for i, line in enumerate(f_src):
                if line == '' or line == '\n':
                    print('\t', "[-]", f"skip line {i+1} of {src_file}")
                    continue
                f_des.write(line)
                total += 1
    print(total, src_file)

def split_datasets():
    split_dict = {}
    target_files = list_iteratively()
    for split in ['train', 'valid', 'test']:
        split_dict[split] = []
        list_path = f"/home/xin/Documents/project/binary-semantics/stateformer/dataset/func_name_cfg/{split}_list.txt"
        des_path = f"/home/xin/Documents/project/binary-semantics/parse_stateformer/dataset_sample/{split}"
        if os.path.exists(des_path):
            shutil.rmtree(des_path)
        if not os.path.exists(des_path):
            os.makedirs(des_path)
        for folder in ['internal_callee1', 'external_callee1', 'internal_callee2', 'external_callee2', 'caller2', 'self', 'caller1']:
            os.makedirs(os.path.join(des_path, folder))
        with open(list_path, 'r') as f:
            for line in f:
                line = line.strip('\n')
                # paths = line.split('/')
                # split_dict[split].append(line)
                src_path = f"/home/xin/Documents/project/binary-semantics/parse_stateformer/sample_dataset/{line}"
                if not os.path.exists(src_path):
                    continue
                for target_file in target_files:
                    src_file = os.path.join(src_path, target_file)
                    des_file = os.path.join(des_path, target_file)
                    copy_lines(src_file, des_file)              
        # assert that all files have the same number of lines
        num_lines = get_num_lines(os.path.join(des_path, 'self', f'input.label'))
        # print(num_lines)
        
        for file in target_files:
            current_num_lines = get_num_lines(os.path.join(des_path, file))
            assert current_num_lines == num_lines, f"number of lines in files are not the same: \n\t {os.path.join(des_path, file)}: {current_num_lines} \n\t {os.path.join(des_path, 'self', f'input.label')}: {num_lines}" 

def get_num_lines(file_name):
    with open(file_name) as f:
        return sum(1 for _ in f)

def list_iteratively():
    result_folder = "sample_output/bc"
    files = glob.glob(os.path.join(result_folder, '*'))
    res = []
    while len(files) > 0:
        file = files.pop()
        if os.path.isdir(file):
            # print(file)
            if not file.endswith('/'):
                file = file + '/'
            new_files = glob.glob(file + '*')
            files = files + new_files
        else:
            if 'input.' in file:
                print(file.replace(result_folder+'/', ''))
                res.append(file.replace(result_folder+'/', ''))
    # print(res)
    return res                

if __name__ == "__main__":
    # test_one_project()
    # main()   
    # main()
    # finished = get_finished()
    # list_iteratively()
    split_datasets()