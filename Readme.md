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

## CAEMO Test
1. week data(*/nfs_baoding_ai/liudong_2023/Monomer/CAEMO_test/pdb_model/quality_estimation_1_week*): 
   * *(project-path)*/CAEMO_test/step_script/1_run_qa_week.sh
   >result: The output result csv file is in (project-path)/CAEMO_test/week_out.csv. Please copy the corresponding file to the corresponding folder.
2. month data(*/nfs_baoding_ai/liudong_2023/Monomer/CAEMO_test/pdb_model/quality_estimation_1_month*):
    * *(project-path)*/CAEMO_test/step_script/1_run_qa_month.sh
    >result: The output result csv file is in (project-path)/CAEMO_test/month_out.csv. Please copy the corresponding file to the corresponding folder.

## Training 
*(project-path)*/bin/train/run_train.sh *project-path* *pdbs-dirs* 
>pdbs-dirs: the folder containing the training data(pdbs). Details see supplement.
