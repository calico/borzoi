#!/bin/bash

# merge bigwig replicates, generate .w5 files and run qc

# define ENCODE ID
ENC_ID='ENCSR000AEL'

# define ENCODE file IDs
FILE_P_REP1='ENCFF980ZHM'
FILE_M_REP1='ENCFF533LJF'

FILE_P_REP2='ENCFF335LVS'
FILE_M_REP2='ENCFF257NOL'

# create folder for merged replicate files
mkdir -p "human/rna/encode/$ENC_ID/summary"


# step 1: generate per-replicate .w5 files

# rep1
if [ -f "human/rna/encode/$ENC_ID/rep1/$FILE_P_REP1+.w5" ]; then
  echo "example RNA-seq .w5 already exists (rep 1)."
else
  bw_h5.py -z "human/rna/encode/$ENC_ID/rep1/$FILE_P_REP1.bigWig" "human/rna/encode/$ENC_ID/rep1/$FILE_P_REP1+.w5"
  bw_h5.py -z "human/rna/encode/$ENC_ID/rep1/$FILE_M_REP1.bigWig" "human/rna/encode/$ENC_ID/rep1/$FILE_M_REP1-.w5"
fi

# rep2
if [ -f "human/rna/encode/$ENC_ID/rep2/$FILE_P_REP2+.w5" ]; then
  echo "example RNA-seq .w5 already exists (rep 2)."
else
  bw_h5.py -z "human/rna/encode/$ENC_ID/rep2/$FILE_P_REP2.bigWig" "human/rna/encode/$ENC_ID/rep2/$FILE_P_REP2+.w5"
  bw_h5.py -z "human/rna/encode/$ENC_ID/rep2/$FILE_M_REP2.bigWig" "human/rna/encode/$ENC_ID/rep2/$FILE_M_REP2-.w5"
fi


# step 2: merge replicates

if [ -f "human/rna/encode/$ENC_ID/summary/coverage+.w5" ]; then
  echo "example RNA-seq .w5 already exists (merged)."
else
  w5_merge.py -w -s mean -z "human/rna/encode/$ENC_ID/summary/coverage+.w5" "human/rna/encode/$ENC_ID/rep1/$FILE_P_REP1+.w5" "human/rna/encode/$ENC_ID/rep2/$FILE_P_REP2+.w5"
  w5_merge.py -w -s mean -z "human/rna/encode/$ENC_ID/summary/coverage-.w5" "human/rna/encode/$ENC_ID/rep1/$FILE_M_REP1-.w5" "human/rna/encode/$ENC_ID/rep2/$FILE_M_REP2-.w5"
fi


# step 3: run qc on each replicate and the merged file

if [ -f "human/rna/encode/$ENC_ID/summary/covqc/means.txt" ]; then
  echo "qc statistics already exist."
else
  # rep1
  w5_qc.py -b "$BORZOI_HG38/blacklist/blacklist_hg38_all.bed" -o "human/rna/encode/$ENC_ID/rep1/covqc" "human/rna/encode/$ENC_ID/rep1/$FILE_P_REP1+.w5"
  w5_qc.py -b "$BORZOI_HG38/blacklist/blacklist_hg38_all.bed" -o "human/rna/encode/$ENC_ID/rep1/covqc_m" "human/rna/encode/$ENC_ID/rep1/$FILE_M_REP1-.w5"

  # rep2
  w5_qc.py -b "$BORZOI_HG38/blacklist/blacklist_hg38_all.bed" -o "human/rna/encode/$ENC_ID/rep2/covqc" "human/rna/encode/$ENC_ID/rep2/$FILE_P_REP2+.w5"
  w5_qc.py -b "$BORZOI_HG38/blacklist/blacklist_hg38_all.bed" -o "human/rna/encode/$ENC_ID/rep2/covqc_m" "human/rna/encode/$ENC_ID/rep2/$FILE_M_REP2-.w5"

  # summary
  w5_qc.py -b "$BORZOI_HG38/blacklist/blacklist_hg38_all.bed" -o "human/rna/encode/$ENC_ID/summary/covqc" "human/rna/encode/$ENC_ID/summary/coverage+.w5"
  w5_qc.py -b "$BORZOI_HG38/blacklist/blacklist_hg38_all.bed" -o "human/rna/encode/$ENC_ID/summary/covqc_m" "human/rna/encode/$ENC_ID/summary/coverage-.w5"
fi

