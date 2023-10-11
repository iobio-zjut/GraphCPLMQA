# Monomer Model Quality Assessment
Model quality assessment of monomeric protein structures using language models.

## With Docker
image:
harbor.biomap-int.com/guangxing_2023/modelselection:v1

## weight file
weight file for network:*xxxx/best.pkl*

weight file for esm_if:*https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt*

weight file for esm2:*https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt*
> Please copy the corresponding file to the corresponding folder. The best.pkl file is stored in *(project-path)*/QA_Model. The weight files of esm_if and esm2 are stored in *(project-path)*/esm-main/esm-pt/

## Predict
*(project-path)*/bin/run_monomer.sh *pdb-dir*
>*pdb-dir: This folder contains the pdb files that need to be evaluated.*
