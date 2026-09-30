#!/usr/bin/env python
# Copyright 2019 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from optparse import OptionParser, OptionGroup
import json
import os
import shutil

import slurm

"""
borzoi_train_folds.py

Train baskerville-tf model replicates on cross folds using given parameters and data.
"""


################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] <params_file> <data1_dir> ..."
    parser = OptionParser(usage)

    # train
    train_options = OptionGroup(parser, "hound_train.py options")
    train_options.add_option(
        "-k",
        dest="keras_fit",
        default=False,
        action="store_true",
        help="Train with Keras fit method [Default: %default]",
    )
    train_options.add_option(
        "-m",
        dest="mixed_precision",
        default=False,
        action="store_true",
        help="Train with mixed precision [Default: %default]",
    )
    train_options.add_option(
        "-o",
        dest="out_dir",
        default="train_out",
        help="Training output directory [Default: %default]",
    )
    train_options.add_option(
        "--restore",
        dest="restore",
        help="Restore model and continue training, from existing fold train dir [Default: %default]",
    )
    train_options.add_option(
        "--trunk",
        dest="trunk",
        default=False,
        action="store_true",
        help="Restore only model trunk [Default: %default]",
    )
    parser.add_option_group(train_options)

    # multi
    rep_options = OptionGroup(parser, "replication options")
    rep_options.add_option(
        "-c",
        dest="crosses",
        default=1,
        type="int",
        help="Number of cross-fold rounds [Default:%default]",
    )
    rep_options.add_option(
        "--checkpoint",
        dest="checkpoint",
        default=False,
        action="store_true",
        help="Restart training from checkpoint [Default: %default]",
    )
    rep_options.add_option(
        "-e",
        dest="conda_env",
        default="tf12",
        help="Anaconda environment [Default: %default]",
    )
    rep_options.add_option(
        "-f",
        dest="fold_subset",
        default=None,
        type="int",
        help="Run a subset of folds [Default:%default]",
    )
    rep_options.add_option(
        "--f_list",
        dest="fold_subset_list",
        default=None,
        help="Run a subset of folds (encoded as comma-separated string) [Default:%default]",
    )
    rep_options.add_option(
        "--identical_crosses",
        dest="identical_crosses",
        default=False,
        action="store_true",
        help="Force all crosses to use the same validation fold [Default: %default]",
    )
    rep_options.add_option(
        "--name",
        dest="name",
        default="fold",
        help="SLURM name prefix [Default: %default]",
    )
    rep_options.add_option(
        "-p",
        dest="processes",
        default=None,
        type="int",
        help="Number of processes, passed by multi script",
    )
    rep_options.add_option(
        "-q",
        dest="queue",
        default="titan_rtx",
        help="SLURM queue on which to run the jobs [Default: %default]",
    )
    rep_options.add_option(
        "-r", "--restart", dest="restart", default=False, action="store_true"
    )
    rep_options.add_option(
        "--setup",
        dest="setup",
        default=False,
        action="store_true",
        help="Setup folds data directory only [Default: %default]",
    )
    parser.add_option_group(rep_options)

    (options, args) = parser.parse_args()

    if len(args) < 2:
        parser.error("Must provide parameters and data directory.")
    else:
        params_file = os.path.abspath(args[0])
        data_dirs = [os.path.abspath(arg) for arg in args[1:]]

    #######################################################
    # prep work

    if not options.restart and os.path.isdir(options.out_dir):
        print(f"Output directory {options.out_dir} exists. Please remove.")
        exit(1)
    os.makedirs(options.out_dir, exist_ok=True)

    # read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
    params_train = params["train"]

    # read data parameters
    num_data = len(data_dirs)
    data_stats_file = f"{data_dirs[0]}/statistics.json"
    with open(data_stats_file) as data_stats_open:
        data_stats = json.load(data_stats_open)

    # count folds
    num_folds = len([dkey for dkey in data_stats if dkey.startswith("fold")])

    # subset folds
    if options.fold_subset is not None:
        num_folds = min(options.fold_subset, num_folds)

    fold_index = [fold_i for fold_i in range(num_folds)]

    # subset folds (list)
    if options.fold_subset_list is not None:
        fold_index = [int(fold_str) for fold_str in options.fold_subset_list.split(",")]

    # arrange data
    for ci in range(options.crosses):
        for fi in fold_index:
            rep_dir = f"{options.out_dir}/f{fi}c{ci}"
            os.makedirs(rep_dir, exist_ok=True)

            # make data directories
            for di in range(num_data):
                rep_data_dir = f"{rep_dir}/data{di}"
                if not os.path.isdir(rep_data_dir):
                    make_rep_data(
                        data_dirs[di], rep_data_dir, fi, ci, options.identical_crosses
                    )

    if options.setup:
        exit(0)

    #######################################################
    # train

    jobs = []

    for ci in range(options.crosses):
        for fi in fold_index:
            rep_dir = f"{options.out_dir}/f{fi}c{ci}"

            train_dir = f"{rep_dir}/train"
            if options.restart and not options.checkpoint and os.path.isdir(train_dir):
                print(f"{rep_dir} found and skipped.")

            else:
                # collect data directories
                rep_data_dirs = [f"{rep_dir}/data{di}" for di in range(num_data)]

                # copy params into output directory
                shutil.copy(params_file, f"{rep_dir}/params.json")

                # train command
                cmd = (
                    (". %s; " % os.environ["BORZOI_CONDA"])
                    if "BORZOI_CONDA" in os.environ
                    else ""
                )
                cmd += "conda activate %s;" % options.conda_env
                cmd += " echo $HOSTNAME;"

                cmd += " hound_train.py"
                cmd += " %s" % options_string(options, train_options, rep_dir)
                cmd += " %s/params.json %s" % (rep_dir, " ".join(rep_data_dirs))

                name = f"{options.name}-train-f{fi}c{ci}"
                sbf = os.path.abspath(f"{rep_dir}/train.sb")
                outf = os.path.abspath(f"{rep_dir}/train.out")
                errf = os.path.abspath(f"{rep_dir}/train.err")

                j = slurm.Job(
                    cmd,
                    name,
                    outf,
                    errf,
                    sbf,
                    queue=options.queue,
                    cpu=8,
                    gpu=params_train.get("num_gpu", 1),
                    mem=30000,
                    time="60-0:0:0",
                )
                jobs.append(j)

    slurm.multi_run(
        jobs, max_proc=options.processes, verbose=True, launch_sleep=10, update_sleep=60
    )


def make_rep_data(data_dir, rep_data_dir, fi, ci, identical_crosses):
    # read data parameters
    data_stats_file = f"{data_dir}/statistics.json"
    with open(data_stats_file) as data_stats_open:
        data_stats = json.load(data_stats_open)

    # sequences per fold
    fold_seqs = []
    dfi = 0
    fold_label = f"fold{dfi}_seqs"
    while fold_label in data_stats:
        fold_seqs.append(data_stats[fold_label])
        del data_stats[fold_label]
        dfi += 1
        fold_label = f"fold{dfi}_seqs"
    num_folds = dfi

    # split folds into train/valid/test
    test_fold = fi
    valid_fold = (fi + 1 + ci) % num_folds
    if identical_crosses:
        valid_fold = (fi + 1) % num_folds
    train_folds = [
        fold for fold in range(num_folds) if fold not in [valid_fold, test_fold]
    ]

    # clear existing directory
    if os.path.isdir(rep_data_dir):
        shutil.rmtree(rep_data_dir)

    # make data directory
    os.makedirs(rep_data_dir, exist_ok=True)

    # dump data stats
    data_stats["test_seqs"] = fold_seqs[test_fold]
    data_stats["valid_seqs"] = fold_seqs[valid_fold]
    data_stats["train_seqs"] = sum([fold_seqs[tf] for tf in train_folds])
    with open(f"{rep_data_dir}/statistics.json", "w") as data_stats_open:
        json.dump(data_stats, data_stats_open, indent=4)

    # set sequence tvt
    seqs_bed_out = open(f"{rep_data_dir}/sequences.bed", "w")
    for line in open(f"{data_dir}/sequences.bed"):
        a = line.split()
        sfi = int(a[-1].replace("fold", ""))
        if sfi == test_fold:
            a[-1] = "test"
        elif sfi == valid_fold:
            a[-1] = "valid"
        else:
            a[-1] = "train"
        print("\t".join(a), file=seqs_bed_out)
    seqs_bed_out.close()

    # copy targets
    shutil.copy(f"{data_dir}/targets.txt", f"{rep_data_dir}/targets.txt")

    # sym link tfrecords
    data_examples_dir = f"{data_dir}/examples"
    rep_examples_dir = f"{rep_data_dir}/examples"
    os.mkdir(rep_examples_dir)

    # test examples
    data_test_dir = f"{data_examples_dir}/fold{test_fold}.zarr"
    rep_test_dir = f"{rep_examples_dir}/test.zarr"
    os.symlink(data_test_dir, rep_test_dir)

    # valid examples
    data_valid_dir = f"{data_examples_dir}/fold{valid_fold}.zarr"
    rep_valid_dir = f"{rep_examples_dir}/valid.zarr"
    os.symlink(data_valid_dir, rep_valid_dir)

    # train examples
    for tfi in train_folds:
        data_train_dir = f"{data_examples_dir}/fold{tfi}.zarr"
        rep_train_dir = f"{rep_examples_dir}/train{tfi}.zarr"
        os.symlink(data_train_dir, rep_train_dir)


def options_string(options, train_options, rep_dir):
    options_str = ""

    for opt in train_options.option_list:
        opt_str = opt.get_opt_string()
        opt_value = options.__dict__[opt.dest]

        # wrap askeriks in ""
        if isinstance(opt_value, str) and opt_value.find("*") != -1:
            opt_value = '"%s"' % opt_value

        # no value for bools
        elif isinstance(opt_value, bool):
            if not opt_value:
                opt_str = ""
            opt_value = ""

        # skip Nones
        elif opt_value is None:
            opt_str = ""
            opt_value = ""

        # modify
        elif opt.dest == "out_dir":
            opt_value = "%s/train" % rep_dir

        # find matching restore
        elif opt.dest == "restore":
            fold_dir_mid = rep_dir.split("/")[-1]
            if options.trunk:
                opt_value = "%s/%s/train/model_trunk.h5" % (opt_value, fold_dir_mid)
            else:
                opt_value = "%s/%s/train/model_best.h5" % (opt_value, fold_dir_mid)

        options_str += " %s %s" % (opt_str, opt_value)

    return options_str


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
