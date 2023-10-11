# Monomer Model Quality Assessment
Model quality assessment of monomeric protein structures using language models.

## With Docker
image:
harbor.biomap-int.com/guangxing_2023/modelselection:v1

## weight file
weight file for network:*/nfs_baoding_ai/liudong_2023/Monomer/QA_Model/best.pkl*

weight file for esm_if:*/nfs_baoding_ai/liudong_2023/Monomer/esm-main/esm-pt/esm_if1_gvp4_t16_142M_UR50.pt*

weight file for esm2:*/nfs_baoding_ai/liudong_2023/Monomer/esm-main/esm-pt/esm2_t33_650M_UR50D.pt*
> Please copy the corresponding file to the corresponding folder.

## Predict
*(project-path)*/bin/run_monomer.sh *pdb-dir*
>*pdb-dir: This folder contains the pdb files that need to be evaluated.*
