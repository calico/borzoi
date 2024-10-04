#!/bin/sh

mkdir -p snp_sad/f3c0

borzoi_sad.py -o snp_sad/f3c0 --rc --stats logD2 -u -t ../../../examples/targets_human.txt ../../../examples/params_pred.json ../../../examples/saved_models/f3c0/train/model0_best.h5 snps_expr.vcf
