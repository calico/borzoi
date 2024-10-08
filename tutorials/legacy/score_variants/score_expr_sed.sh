#!/bin/sh

mkdir -p snp_sed/f3c0

borzoi_sed.py -o snp_sed/f3c0 --rc --stats logSED,logD2 -u -t ../../../examples/targets_gtex.txt ../../../examples/params_pred.json ../../../examples/saved_models/f3c0/train/model0_best.h5 snps_expr.vcf
