#!/bin/bash
input_dir="$1"
input_emb="$1/seq_emb"
input_str_emb="$1/seq_s_emb"
input_fa="$1/all.fa"
project_path=$(dirname "$(dirname "$0")")
echo "work dir:$input_dir"
del_temp_file="T"

# Process files
if [ $# -ge 2 ]; then
    del_temp_file="$2"
fi

if [ $del_temp_file == "T" ]; then
   echo "Process files are not kept."
elif [ $del_temp_file == "F" ]; then
   echo "Process files are kept."
else
   echo "There is a problem with the entered string."
fi

## run seq emb ###
if [ ! -f $input_fa ]; then
  echo "doing generate fastas ..."
  python $project_path/QA_File/QA_utils/mul_fa.py $input_dir
fi

if [ ! -d $input_emb ]; then
  mkdir $input_emb
fi
python $project_path/esm-main/monomer_seq_emb.py $input_fa $input_emb --include mean --bv 2e1

### run structure emb ###
if [ ! -d $input_str_emb ]; then
  mkdir $input_str_emb
fi
python $project_path/esm-main/monomer_str_emb.py $input_dir $input_str_emb

## run QA ###
cd $project_path
python $project_path/bin/Monomer_QA.py -lt -v -p 20 $input_dir $input_str_emb $input_dir

# check 
if [ $del_temp_file == "T" ]; then
   rm -rf $input_dir/seq_emb
   rm -rf $input_dir/seq_s_emb
   rm -rf $input_dir/*.features.npz
   rm -rf $input_dir/*.fa
fi