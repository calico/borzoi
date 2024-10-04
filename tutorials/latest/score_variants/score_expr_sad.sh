#!/bin/sh

mkdir -p snp_sad/f0c0

borzoi_sad.py -o snp_sad/f0c0 --rc --stats logD2 -t ../make_data/targets_human.txt ../train_model/params_mini.json ../train_model/mini_models/f0c0/train/model_best.h5 snps_expr.vcf
