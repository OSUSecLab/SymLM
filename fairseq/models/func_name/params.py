fields = ['static', 'inst_pos_emb', 'op_pos_emb', 'arch_emb', 'byte1', 'byte2', 'byte3', 'byte4']
opts = ['O0', 'O1', 'O2', 'O3']
field_cf = 'cover'
byte_start_pos = 4

output_langs = ['byte1', 'byte2', 'byte3', 'byte4']
trace_weights = {'byte1': 0.25, 'byte2': 0.25, 'byte3': 0.25, 'byte4': 0.25}
