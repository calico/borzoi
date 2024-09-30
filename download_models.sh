#!/bin/bash

# download model weights (data fold 3, 4 replicates)
for rep in f3c0,f0 f3c1,f1 f3c2,f2 f3c3,f3; do IFS=","; set -- $rep; 
  mkdir -p "examples/saved_models/$1/train"
  local_model="examples/saved_models/$1/train/model0_best.h5"
  if [ -f "$local_model" ]; then
    echo "$1 model already exists."
  else
    wget --progress=bar:force "https://storage.googleapis.com/seqnn-share/borzoi/$2/model0_best.h5" -O "$local_model"
  fi
done

# download and uncompress annotation files
mkdir -p examples/hg38/genes/gencode41
mkdir -p examples/hg38/genes/polyadb

if [ -f examples/hg38/genes/gencode41/gencode41_basic_nort.gtf ]; then
  echo "Gene annotation already exists."
else
  wget -O - https://storage.googleapis.com/seqnn-share/helper/gencode41_basic_nort.gtf.gz | gunzip -c > examples/hg38/genes/gencode41/gencode41_basic_nort.gtf
fi

if [ -f examples/hg38/genes/gencode41/gencode41_basic_nort_protein.gtf ]; then
  echo "Gene annotation (no read-through, protein-coding) already exists."
else
  wget -O - https://storage.googleapis.com/seqnn-share/helper/gencode41_basic_nort_protein.gtf.gz | gunzip -c > examples/hg38/genes/gencode41/gencode41_basic_nort_protein.gtf
fi

if [ -f examples/hg38/genes/gencode41/gencode41_basic_protein.gtf ]; then
  echo "Gene annotation (protein-coding) already exists."
else
  wget -O - https://storage.googleapis.com/seqnn-share/helper/gencode41_basic_protein.gtf.gz | gunzip -c > examples/hg38/genes/gencode41/gencode41_basic_protein.gtf
fi

if [ -f examples/hg38/genes/gencode41/gencode41_basic_tss2.bed ]; then
  echo "TSS annotation already exists."
else
  wget -O - https://storage.googleapis.com/seqnn-share/helper/gencode41_basic_tss2.bed.gz | gunzip -c > examples/hg38/genes/gencode41/gencode41_basic_tss2.bed
fi

if [ -f examples/hg38/genes/gencode41/gencode41_basic_protein_splice.csv.gz ]; then
  echo "Splice site annotation already exist."
else
  wget https://storage.googleapis.com/seqnn-share/helper/gencode41_basic_protein_splice.csv.gz -O examples/hg38/genes/gencode41/gencode41_basic_protein_splice.csv.gz
fi

if [ -f examples/hg38/genes/gencode41/gencode41_basic_protein_splice.gff ]; then
  echo "Splice site annotation already exist."
else
  wget -O - https://storage.googleapis.com/seqnn-share/helper/gencode41_basic_protein_splice.gff.gz | gunzip -c > examples/hg38/genes/gencode41/gencode41_basic_protein_splice.gff
fi

if [ -f examples/hg38/genes/polyadb/polyadb_human_v3.csv.gz ]; then
  echo "PolyA site annotation already exist."
else
  wget https://storage.googleapis.com/seqnn-share/helper/polyadb_human_v3.csv.gz -O examples/hg38/genes/polyadb/polyadb_human_v3.csv.gz
fi

# download and index hg38 genome
mkdir -p examples/hg38/assembly/ucsc

if [ -f examples/hg38/assembly/ucsc/hg38.fa ]; then
  echo "Human genome FASTA already exists."
else
  wget -O - http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz | gunzip -c > examples/hg38/assembly/ucsc/hg38.fa
  python src/scripts/idx_genome.py examples/hg38/assembly/ucsc/hg38.fa
fi
