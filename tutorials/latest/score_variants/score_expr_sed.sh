#!/bin/sh

mkdir -p snp_sed/f0c0

borzoi_sed.py -o snp_sed/f0c0 --rc --stats logSED,logD2 -t ../make_data/targets_human.txt ../train_model/params_mini.json ../train_model/mini_models/f0c0/train/model_best.h5 snps_expr.vcf
