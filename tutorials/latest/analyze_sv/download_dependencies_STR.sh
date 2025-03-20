#!/bin/bash

# create additional folder in borzoi data folders
mkdir -p "data/model"
mkdir -p "data/model/f0"
mkdir -p "data/model/f1"
mkdir -p "data/model/f2"
mkdir -p "data/model/f3"

# download dependencies and the model
if [ -f "data/hg19.fa" ]; then
  echo "hg19.fa already exists."
else
  wget -O - "ftp://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz" | gunzip -c > "data/hg19.fa"
fi

if [ -f "data/gencode41_lift37_exons.bed" ]; then
  echo "gencode41_lift37_exons.bed already exists."
else
  wget -O - "https://storage.googleapis.com/seqnn-share/helper/gencode41_lift37_exons.bed.gz" | gunzip -c > "data/gencode41_lift37_exons.bed"
fi

if [ -f "data/model/f0/model0_best.h5" ]; then
  echo "f0/model0_best.h5 already exists."
else
  wget "https://storage.googleapis.com/seqnn-share/borzoi/f0/model0_best.h5" -O "data/model/f0/model0_best.h5"
fi

if [ -f "data/model/f1/model0_best.h5" ]; then
  echo "f1/model0_best.h5 already exists."
else
  wget "https://storage.googleapis.com/seqnn-share/borzoi/f1/model0_best.h5" -O "data/model/f1/model0_best.h5"
fi

if [ -f "data/model/f2/model0_best.h5" ]; then
  echo "f2/model0_best.h5 already exists."
else
  wget "https://storage.googleapis.com/seqnn-share/borzoi/f2/model0_best.h5" -O "data/model/f2/model0_best.h5"
fi

if [ -f "data/model/f3/model0_best.h5" ]; then
  echo "f3/model0_best.h5 already exists."
else
  wget "https://storage.googleapis.com/seqnn-share/borzoi/f3/model0_best.h5" -O "data/model/f3/model0_best.h5"
fi

if [ -f "data/targets.txt" ]; then
  echo "targets.txt already exists."
else
  wget "https://storage.googleapis.com/seqnn-share/borzoi/hg38/targets.txt" -O "data/targets.txt"
fi

if [ -f "data/params.json" ]; then
  echo "params.json already exists."
else
  wget "https://storage.googleapis.com/seqnn-share/borzoi/params.json" -O "data/params.json"
fi
