#!/bin/sh

mkdir -p snp_polya/f3c0

borzoi_sed_paqtl_cov.py -o snp_polya/f3c0 --rc --stats COVR -u -t ../../../examples/targets_rna.txt ../../../examples/params_pred.json ../../../examples/saved_models/f3c0/train/model0_best.h5 snps_polya.vcf
