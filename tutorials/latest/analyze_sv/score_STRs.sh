#!bin/bash

#python save_STR_vcf.py --input data/STR.csv \
#    --fasta data/hg19.fa \
#    --output_dir data/vcfs_STR

python score_tandem_repeats.py --table data/STR.csv \
    --input data/vcfs_STR \
    --fasta data/hg19.fa \
    --model data/model \
    --params data/params.json \
    --targets data/targets.txt \
    --gencode data/gencode41_lift37_exons.bed \
    --output_dir out_STR \
