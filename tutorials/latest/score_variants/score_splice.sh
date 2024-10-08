#!/bin/sh

mkdir -p snp_splice/f0c0

borzoi_sed.py -o snp_splice/f0c0 --span --no_untransform --rc --stats nDi -t ../make_data/targets_human.txt ../train_model/params_mini.json ../train_model/mini_models/f0c0/train/model_best.h5 snps_splice.vcf
