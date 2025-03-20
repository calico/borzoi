#!bin/bash

python analyze_vcf.py --vcf data/chr6_41897087_SV.vcf \
    --fasta data/hg38.fa \
    --model data/model \
    --params data/params.json \
    --targets data/targets.txt \
    --gencode data/gencode41_basic_exons.bed \
    --output_dir temp \
    --fig_width 1000
