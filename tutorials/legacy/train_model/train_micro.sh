#!/bin/sh

westminster_train_folds.py -e borzoi_py310 -f 2 -c 1 -q rtx4090 -o micro_models params_micro.json ../make_data/data/hg38
