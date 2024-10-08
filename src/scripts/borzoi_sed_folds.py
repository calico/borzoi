#!/usr/bin/env python
# Copyright 2019 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#         https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from optparse import OptionParser, OptionGroup
import glob
import h5py
import json
import pdb
import os
import sys

import numpy as np
import pandas as pd

import slurm

"""
borzoi_sed_folds.py

Compute SED scores across model folds.
"""

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <data_dir>'
    parser = OptionParser(usage)

    # sed
    sed_options = OptionGroup(parser, 'borzoi_sed_folds.py options')
    sed_options.add_option(
        '-f',
        dest='genome_fasta',
        default='%s/assembly/ucsc/hg38.fa' % os.environ.get('BORZOI_HG38', 'hg38'),
        help='Genome FASTA for sequences [Default: %default]',
    )
    sed_options.add_option(
        '-g',
        dest='genes_gtf',
        default='%s/genes/gencode41/gencode41_basic_nort.gtf' % os.environ.get('BORZOI_HG38', 'hg38'),
        help='GTF for gene definition [Default %default]',
    )
    sed_options.add_option(
        '-o',
        dest='out_dir',
        default='sed',
        help='Output directory for tables and plots [Default: %default]',
    )
    sed_options.add_option(
        '-p',
        dest='processes',
        default=None,
        type='int',
        help='Number of processes, passed by multi script',
    )
    sed_options.add_option(
        '--rc',
        dest='rc',
        default=False,
        action='store_true',
        help='Average forward and reverse complement predictions [Default: %default]',
    )
    sed_options.add_option(
        '--shifts',
        dest='shifts',
        default='0',
        type='str',
        help='Ensemble prediction shifts [Default: %default]',
    )
    sed_options.add_option(
        '--span',
        dest='span',
        default=False,
        action='store_true',
        help='Aggregate entire gene span [Default: %default]',
    )
    sed_options.add_option(
        '--stats',
        dest='sed_stats',
        default='SED',
        help='Comma-separated list of stats to save. [Default: %default]',
    )
    sed_options.add_option(
        '-t',
        dest='targets_file',
        default=None,
        type='str',
        help='File specifying target indexes and labels in table format',
    )
    sed_options.add_option(
        '-u',
        dest='untransform_old',
        default=False,
        action='store_true',
    )
    sed_options.add_option(
        '--no_untransform',
        dest='no_untransform',
        default=False,
        action='store_true',
    )
    parser.add_option_group(sed_options)
    
    # cross-fold
    fold_options = OptionGroup(parser, 'cross-fold options')
    fold_options.add_option(
        '-c',
        dest='crosses',
        default=1,
        type='int',
        help='Number of cross-fold rounds [Default:%default]',
    )
    fold_options.add_option(
        '--folds',
        dest='fold_subset',
        default=1,
        type='int',
        help='Run a subset of folds [Default:%default]',
    )
    fold_options.add_option(
        '--f_list',
        dest='fold_subset_list',
        default=None,
        help='Run a subset of folds (encoded as comma-separated string) [Default:%default]',
    )
    fold_options.add_option(
        '-d',
        dest='data_head',
        default=None,
        type='int',
        help='Index for dataset/head [Default: %default]',
    )
    fold_options.add_option(
        '-e',
        dest='conda_env',
        default='tf210',
        help='Anaconda environment [Default: %default]',
    )
    fold_options.add_option(
        '--name',
        dest='name',
        default='sed',
        help='SLURM name prefix [Default: %default]',
    )
    fold_options.add_option(
        '--max_proc',
        dest='max_proc',
        default=None,
        type='int',
        help='Maximum concurrent processes [Default: %default]',
    )
    fold_options.add_option(
        '-q',
        dest='queue',
        default='geforce',
        help='SLURM queue on which to run the jobs [Default: %default]',
    )
    fold_options.add_option(
        '-r',
        dest='restart',
        default=False,
        action='store_true',
        help='Restart a partially completed job [Default: %default]',
    )
    fold_options.add_option(
        '--vcf',
        dest='vcf_file',
        default='/home/drk/seqnn/data/gtex_fine/susie_pip90/pos_merge.vcf',
    )
    parser.add_option_group(fold_options)

    (options, args) = parser.parse_args()

    if len(args) != 2:
        parser.error('Must provide parameters file and cross-fold directory')
    else:
        params_file = args[0]
        exp_dir = args[1]

    #######################################################
    # prep work

    # set folds
    num_folds = 1
    if options.fold_subset is not None:
        num_folds = options.fold_subset
  
    fold_index = [fold_i for fold_i in range(num_folds)]

    # subset folds (list)
    if options.fold_subset_list is not None:
        fold_index = [int(fold_str) for fold_str in options.fold_subset_list.split(",")]

    ################################################################
    # SNP scores

    # command base
    cmd_base = ('. %s; ' % os.environ['BORZOI_CONDA']) if 'BORZOI_CONDA' in os.environ else ''
    cmd_base += 'conda activate %s;' % options.conda_env
    cmd_base += ' echo $HOSTNAME;'

    jobs = []

    for ci in range(options.crosses):
        for fi in fold_index:
            it_dir = '%s/f%dc%d' % (exp_dir, fi, ci)
            name = '%s-f%dc%d' % (options.name, fi, ci)

            # update output directory
            it_out_dir = '%s/%s' % (it_dir, options.out_dir)
            os.makedirs(it_out_dir, exist_ok=True)

            model_file = '%s/train/model_best.h5' % it_dir
            if options.data_head is not None:
                model_file = '%s/train/model%d_best.h5' % (it_dir, options.data_head)

            cmd_fold = '%s time borzoi_sed.py %s %s' % (cmd_base, params_file, model_file)

            # variant scoring job
            job_out_dir = it_out_dir
            if not options.restart or not os.path.isfile('%s/sed.h5'%job_out_dir):
                cmd_job = '%s %s' % (cmd_fold, options.vcf_file)
                cmd_job += ' %s' % options_string(options, sed_options, job_out_dir)
                j = slurm.Job(cmd_job, '%s' % name,
                        '%s.out'%job_out_dir, '%s.err'%job_out_dir, '%s.sb'%job_out_dir,
                        queue=options.queue, gpu=1,
                        mem=60000, time='30-0:0:0')
                jobs.append(j)
                
    slurm.multi_run(jobs, max_proc=options.max_proc, verbose=True,
                                    launch_sleep=10, update_sleep=60)

def options_string(options, group_options, rep_dir):
    options_str = ''

    for opt in group_options.option_list:
        opt_str = opt.get_opt_string()
        opt_value = options.__dict__[opt.dest]

        # wrap askeriks in ""
        if type(opt_value) == str and opt_value.find('*') != -1:
            opt_value = '"%s"' % opt_value

        # no value for bools
        elif type(opt_value) == bool:
            if not opt_value:
                opt_str = ''
            opt_value = ''

        # skip Nones
        elif opt_value is None:
            opt_str = ''
            opt_value = ''

        # modify
        elif opt.dest == 'out_dir':
            opt_value = rep_dir

        options_str += ' %s %s' % (opt_str, opt_value)

    return options_str

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
