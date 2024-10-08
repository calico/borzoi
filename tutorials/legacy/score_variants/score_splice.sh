#!/bin/sh

mkdir -p snp_splice/f3c0

borzoi_sed.py -o snp_splice/f3c0 --span --no_untransform --rc --stats nDi -u -t ../../../examples/targets_rna.txt ../../../examples/params_pred.json ../../../examples/saved_models/f3c0/train/model0_best.h5 snps_splice.vcf
