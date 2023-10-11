import os
import argparse
import numpy as np
import esm.inverse_folding
import os
import torch

os.environ['MKL_THREADING_LAYER'] = 'GNU'
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("decoys_dirs", type=str, help="decoys dir")
parser.add_argument("out_dirs", type=str, help="decoys dir")

args = parser.parse_args()
decoys_dirs=args.decoys_dirs
out_dirs=args.out_dirs

decoys_dirs_list=os.listdir(decoys_dirs)

script_dir=os.path.abspath(os.path.dirname(__file__))
for i in decoys_dirs_list:
    decoys_dirs_i=os.path.join(decoys_dirs,i)
    out_dirs_i=os.path.join(out_dirs,i)
    if not os.path.exists(out_dirs_i):
        os.makedirs(out_dirs_i)
    os.system("python %s/monomer_str_emb.py %s %s"%(script_dir, decoys_dirs_i, out_dirs_i))
    

