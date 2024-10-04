#!/bin/bash

# create additional folder in borzoi data folders
mkdir -p "$BORZOI_HG38/assembly/ucsc"
mkdir -p "$BORZOI_HG38/assembly/gnomad"
mkdir -p "$BORZOI_HG38/mappability"
mkdir -p "$BORZOI_HG38/blacklist"
mkdir -p "$BORZOI_HG38/align"

mkdir -p "$BORZOI_MM10/assembly/ucsc"
mkdir -p "$BORZOI_MM10/mappability"
mkdir -p "$BORZOI_MM10/blacklist"


# download and uncompress auxiliary files required for Makefile (hg38)
if [ -f "$BORZOI_HG38/assembly/ucsc/hg38_gaps.bed" ]; then
  echo "hg38_gaps.bed already exists."
else
  wget -O - https://storage.googleapis.com/seqnn-share/helper/dependencies/hg38_gaps.bed.gz | gunzip -c > "$BORZOI_HG38/assembly/ucsc/hg38_gaps.bed"
fi

if [ -f "$BORZOI_HG38/mappability/umap_k36_t10_l32.bed" ]; then
  echo "umap_k36_t10_l32.bed (hg38) already exists."
else
  wget -O - https://storage.googleapis.com/seqnn-share/helper/dependencies/umap_k36_t10_l32_hg38.bed.gz | gunzip -c > "$BORZOI_HG38/mappability/umap_k36_t10_l32.bed"
fi

if [ -f "$BORZOI_HG38/blacklist/blacklist_hg38_all.bed" ]; then
  echo "blacklist_hg38_all.bed already exists."
else
  wget -O - https://storage.googleapis.com/seqnn-share/helper/dependencies/blacklist_hg38_all.bed.gz | gunzip -c > "$BORZOI_HG38/blacklist/blacklist_hg38_all.bed"
fi

if [ -f "$BORZOI_HG38/align/hg38.mm10.syn.net.gz" ]; then
  echo "Splice site annotation already exist."
else
  wget https://storage.googleapis.com/seqnn-share/helper/dependencies/hg38.mm10.syn.net.gz -O "$BORZOI_HG38/align/hg38.mm10.syn.net.gz"
fi


# download and uncompress auxiliary files required for Makefile (mm10)
if [ -f "$BORZOI_MM10/assembly/ucsc/mm10_gaps.bed" ]; then
  echo "mm10_gaps.bed already exists."
else
  wget -O - https://storage.googleapis.com/seqnn-share/helper/dependencies/mm10_gaps.bed.gz | gunzip -c > "$BORZOI_MM10/assembly/ucsc/mm10_gaps.bed"
fi

if [ -f "$BORZOI_MM10/mappability/umap_k36_t10_l32.bed" ]; then
  echo "umap_k36_t10_l32.bed (mm10) already exists."
else
  wget -O - https://storage.googleapis.com/seqnn-share/helper/dependencies/umap_k36_t10_l32_mm10.bed.gz | gunzip -c > "$BORZOI_MM10/mappability/umap_k36_t10_l32.bed"
fi

if [ -f "$BORZOI_MM10/blacklist/blacklist_mm10_all.bed" ]; then
  echo "blacklist_mm10_all.bed already exists."
else
  wget -O - https://storage.googleapis.com/seqnn-share/helper/dependencies/blacklist_mm10_all.bed.gz | gunzip -c > "$BORZOI_MM10/blacklist/blacklist_mm10_all.bed"
fi


# download and uncompress pre-compiled umap bed files
if [ -f umap_human.bed ]; then
  echo "umap_human.bed already exists."
else
  wget -O - https://storage.googleapis.com/seqnn-share/helper/dependencies/umap_human.bed.gz | gunzip -c > umap_human.bed
fi

if [ -f umap_mouse.bed ]; then
  echo "umap_mouse.bed already exists."
else
  wget -O - https://storage.googleapis.com/seqnn-share/helper/dependencies/umap_mouse.bed.gz | gunzip -c > umap_mouse.bed
fi


# download and index hg38 ml genome
if [ -f "$BORZOI_HG38/assembly/ucsc/hg38.ml.fa" ]; then
  echo "hg38.ml.fa already exists."
else
  wget -O - https://storage.googleapis.com/seqnn-share/helper/dependencies/hg38.ml.fa.gz | gunzip -c > "$BORZOI_HG38/assembly/ucsc/hg38.ml.fa"
  idx_genome.py "$BORZOI_HG38/assembly/ucsc/hg38.ml.fa"
fi

# download and index hg38 ml genome (gnomad major alleles)
if [ -f "$BORZOI_HG38/assembly/gnomad/hg38.ml.fa" ]; then
  echo "hg38.ml.fa (gnomad) already exists."
else
  wget -O - https://storage.googleapis.com/seqnn-share/helper/dependencies/hg38_gnomad.ml.fa.gz | gunzip -c > "$BORZOI_HG38/assembly/gnomad/hg38.ml.fa"
  idx_genome.py "$BORZOI_HG38/assembly/gnomad/hg38.ml.fa"
fi

# download and index mm10 ml genome
if [ -f "$BORZOI_MM10/assembly/ucsc/mm10.ml.fa" ]; then
  echo "mm10.ml.fa already exists."
else
  wget -O - https://storage.googleapis.com/seqnn-share/helper/dependencies/mm10.ml.fa.gz | gunzip -c > "$BORZOI_MM10/assembly/ucsc/mm10.ml.fa"
  idx_genome.py "$BORZOI_MM10/assembly/ucsc/mm10.ml.fa"
fi
