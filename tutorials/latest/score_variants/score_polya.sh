#!/bin/sh

mkdir -p snp_polya/f0c0

borzoi_sed_paqtl_cov.py -o snp_polya/f0c0 --rc --stats COVR -t ../make_data/targets_human.txt ../train_model/params_mini.json ../train_model/mini_models/f0c0/train/model_best.h5 snps_polya.vcf
