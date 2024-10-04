#!/bin/sh

borzoi_satg_gene.py -o k562_HBE1 -f 0 -c 0 --rc --track_scale 0.3 --track_transform 0.5 --clip_soft 384.0 -t ../make_data/targets_human.txt ../train_model/params_mini.json ../train_model/mini_models HBE1_example.gtf
