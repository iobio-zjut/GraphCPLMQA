# parse pdb
import argparse
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("input_dirs", type=str, help="input dirs")
parser.add_argument("out_dirs", type=str, help="input dirs")
args = parser.parse_args()

input_dirs= args.input_dirs
out_dirs= args.out_dirs

input_dirs_list=os.listdir(input_dirs)
script_dir=os.path.abspath(os.path.dirname(__file__)).replace("train","")

for pdir in input_dirs_list:
    dirs_pdb = os.path.join(input_dirs, pdir)
    if os.path.exists(dirs_pdb):
        out_dirs_i=os.path.join(out_dirs, pdir)
        if not os.path.exists(out_dirs_i):
            os.makedirs(out_dirs_i)
        cmd = "python %sMonomer_QA.py -p 20 -f -v %s %s %s "%(script_dir, dirs_pdb, "train", out_dirs_i)
        print(cmd)
        os.system(cmd)



