# Please read carefully

# Since this data set is very large, 
# there are many feature files required for training the network, 
# so please run it separately if you use step-by-step generation
# see example folder: there are four different targets folders, which contain some decoys for training

project_path=$1
train_pdbs_dir=$2  
data_path=$(dirname "$train_pdbs_dir")

# Step 1.0 generate seq emb  
if [ ! -f $data_path/all.fa ]; then
  echo "doing generate fastas ..."
  python $project_path/bin/train/train_mul_fa.py $train_pdbs_dir
fi


if [ ! -d $data_path/seq_emb ]; then
  mkdir $data_path/seq_emb
fi

python $project_path/esm-main/monomer_seq_emb.py $data_path/all.fa $data_path/seq_emb  --include mean --bv 2e1

# Step 1.1 generate str emb
if [ ! -d $data_path/seq_s_emb ]; then
  mkdir $data_path/seq_s_emb
fi

python $project_path/esm-main/monomer_str_emb_mul.py $train_pdbs_dir $data_path/seq_s_emb

# Step 1.2 generate other features
if [ ! -d $data_path/features ]; then
  mkdir $data_path/features
fi

cd /nfs_baoding_ai/liudong_2023/Monomer
python $project_path/bin/train/train_mul_fea.py $train_pdbs_dir $data_path/features

# Step 2 tarin start
if [ ! -d $data_path/out_ckpt ]; then
  mkdir $data_path/out_ckpt
fi
# cd /nfs_baoding_ai/liudong_2023/Monomer
python /nfs_baoding_ai/liudong_2023/Monomer/bin/train/Train_Monomer.py $data_path $data_path/out_ckpt