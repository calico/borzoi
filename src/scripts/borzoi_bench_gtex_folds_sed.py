#!/usr/bin/env python
# Copyright 2023 Calico LLC

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
import json
import pickle
import pdb
import os
import shutil
import sys

import h5py
import numpy as np
import pandas as pd

import slurm

"""
borzoi_bench_gtex_folds_sed.py

Benchmark Borzoi model replicates on GTEx eQTL coefficient task (gene-specific).
"""

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <data_dir>'
    parser = OptionParser(usage)

    # sed options
    sed_options = OptionGroup(parser, 'borzoi_sed.py options')
    sed_options.add_option(
        '-b',
        dest='bedgraph',
        default=False,
        action='store_true',
        help='Write ref/alt predictions as bedgraph [Default: %default]',
    )
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
    sed_options.add_option(
        "--no_unclip",
        dest="no_unclip",
        default=False,
        action="store_true",
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
        '--gtex',
        dest='gtex_vcf_dir',
        default='/home/drk/seqnn/data/gtex_fine/susie_pip90',
    )
    fold_options.add_option(
        '--susie',
        dest='susie_dir',
        default=None,
    )
    fold_options.add_option(
        '--name',
        dest='name',
        default='gtex',
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
        '-p',
        dest='processes',
        default=None,
        type='int',
        help='Number of processes, passed by multi script.',
    )
    fold_options.add_option(
        '-q',
        dest='queue',
        default='geforce',
        help='SLURM queue on which to run the jobs [Default: %default]',
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

    # extract output subdirectory name
    gtex_out_dir = options.out_dir

    # split SNP stats
    sed_stats = options.sed_stats.split(',')

    # merge study/tissue variants
    mpos_vcf_file = '%s/pos_merge.vcf' % options.gtex_vcf_dir
    mneg_vcf_file = '%s/neg_merge.vcf' % options.gtex_vcf_dir

    ################################################################
    # SED

    # SED command base
    cmd_base = ('. %s; ' % os.environ['BORZOI_CONDA']) if 'BORZOI_CONDA' in os.environ else ''
    cmd_base += 'conda activate %s;' % options.conda_env
    cmd_base += ' echo $HOSTNAME;'

    jobs = []

    for ci in range(options.crosses):
        for fi in fold_index:
            it_dir = '%s/f%dc%d' % (exp_dir, fi, ci)
            name = '%s-f%dc%d' % (options.name, fi, ci)

            # update output directory
            it_out_dir = '%s/%s' % (it_dir, gtex_out_dir)
            os.makedirs(it_out_dir, exist_ok=True)

            # choose model
            model_file = '%s/train/model_best.h5' % it_dir
            if options.data_head is not None:
                model_file = '%s/train/model%d_best.h5' % (it_dir, options.data_head)

            ########################################
            # negative jobs
            
            # pickle options
            options.out_dir = '%s/merge_neg' % it_out_dir
            os.makedirs(options.out_dir, exist_ok=True)
            options_pkl_file = '%s/options.pkl' % options.out_dir
            options_pkl = open(options_pkl_file, 'wb')
            pickle.dump(options, options_pkl)
            options_pkl.close()

            # create base fold command
            cmd_fold = '%s time borzoi_sed.py %s %s %s' % (
                cmd_base, options_pkl_file, params_file, model_file)

            for pi in range(options.processes):
                sed_file = '%s/job%d/sed.h5' % (options.out_dir, pi)
                if not nonzero_h5(sed_file, sed_stats):
                    cmd_job = '%s %s %d' % (cmd_fold, mneg_vcf_file, pi)
                    j = slurm.Job(cmd_job, '%s_neg%d' % (name,pi),
                            '%s/job%d.out' % (options.out_dir,pi),
                            '%s/job%d.err' % (options.out_dir,pi),
                            '%s/job%d.sb' % (options.out_dir,pi),
                            queue=options.queue, gpu=1, cpu=2,
                            mem=48000, time='7-0:0:0')
                    jobs.append(j)

            ########################################
            # positive jobs
            
            # pickle options
            options.out_dir = '%s/merge_pos' % it_out_dir
            os.makedirs(options.out_dir, exist_ok=True)
            options_pkl_file = '%s/options.pkl' % options.out_dir
            options_pkl = open(options_pkl_file, 'wb')
            pickle.dump(options, options_pkl)
            options_pkl.close()

            # create base fold command
            cmd_fold = '%s time borzoi_sed.py %s %s %s' % (
                cmd_base, options_pkl_file, params_file, model_file)

            for pi in range(options.processes):
                sed_file = '%s/job%d/sed.h5' % (options.out_dir, pi)
                if not nonzero_h5(sed_file, sed_stats):
                    cmd_job = '%s %s %d' % (cmd_fold, mpos_vcf_file, pi)
                    j = slurm.Job(cmd_job, '%s_pos%d' % (name,pi),
                            '%s/job%d.out' % (options.out_dir,pi),
                            '%s/job%d.err' % (options.out_dir,pi),
                            '%s/job%d.sb' % (options.out_dir,pi),
                            queue=options.queue, gpu=1, cpu=2,
                            mem=48000, time='7-0:0:0')
                    jobs.append(j)

    slurm.multi_run(jobs, max_proc=options.max_proc, verbose=True,
                                    launch_sleep=10, update_sleep=60)

    #######################################################
    # collect output

    for ci in range(options.crosses):
        for fi in fold_index:
            it_out_dir = '%s/f%dc%d/%s' % (exp_dir, fi, ci, gtex_out_dir)

            # collect negatives
            neg_out_dir = '%s/merge_neg' % it_out_dir
            if not os.path.isfile('%s/sed.h5' % neg_out_dir):
                collect_scores(neg_out_dir, options.processes, 'sed.h5')

            # collect positives
            pos_out_dir = '%s/merge_pos' % it_out_dir
            if not os.path.isfile('%s/sed.h5' % pos_out_dir):
                collect_scores(pos_out_dir, options.processes, 'sed.h5')


    ################################################################
    # split study/tissue variants

    for ci in range(options.crosses):
        for fi in fold_index:
            it_out_dir = '%s/f%dc%d/%s' % (exp_dir, fi, ci, gtex_out_dir)
            print(it_out_dir)

            # split positives
            split_scores(it_out_dir, 'pos', options.gtex_vcf_dir, sed_stats)

            # split negatives
            split_scores(it_out_dir, 'neg', options.gtex_vcf_dir, sed_stats)

    ################################################################
    # ensemble
    
    ensemble_dir = '%s/ensemble' % exp_dir
    if not os.path.isdir(ensemble_dir):
        os.mkdir(ensemble_dir)

    gtex_dir = '%s/%s' % (ensemble_dir, gtex_out_dir)
    if not os.path.isdir(gtex_dir):
        os.mkdir(gtex_dir)

    for gtex_pos_vcf in glob.glob('%s/*_pos.vcf' % options.gtex_vcf_dir):
        gtex_neg_vcf = gtex_pos_vcf.replace('_pos.','_neg.')
        pos_base = os.path.splitext(os.path.split(gtex_pos_vcf)[1])[0]
        neg_base = os.path.splitext(os.path.split(gtex_neg_vcf)[1])[0]

        # collect SED files
        sed_pos_files = []
        sed_neg_files = []
        for ci in range(options.crosses):
            for fi in fold_index:
                it_dir = '%s/f%dc%d' % (exp_dir, fi, ci)
                it_out_dir = '%s/%s' % (it_dir, gtex_out_dir)
                
                sed_pos_file = '%s/%s/sed.h5' % (it_out_dir, pos_base)
                sed_pos_files.append(sed_pos_file)

                sed_neg_file = '%s/%s/sed.h5' % (it_out_dir, neg_base)
                sed_neg_files.append(sed_neg_file)

        # ensemble
        ens_pos_dir = '%s/%s' % (gtex_dir, pos_base)
        os.makedirs(ens_pos_dir, exist_ok=True)
        ens_pos_file = '%s/sed.h5' % (ens_pos_dir)
        if not os.path.isfile(ens_pos_file):
            ensemble_h5(ens_pos_file, sed_pos_files, sed_stats)

        ens_neg_dir = '%s/%s' % (gtex_dir, neg_base)
        os.makedirs(ens_neg_dir, exist_ok=True)
        ens_neg_file = '%s/sed.h5' % (ens_neg_dir)
        if not os.path.isfile(ens_neg_file):
            ensemble_h5(ens_neg_file, sed_neg_files, sed_stats)


    ################################################################
    # coefficient analysis

    if options.susie_dir is not None :
        cmd_base = 'borzoi_gtex_coef_sed.py -g %s --susie %s' % (options.gtex_vcf_dir, options.susie_dir)

        jobs = []
        for ci in range(options.crosses):
            for fi in fold_index:
                it_dir = '%s/f%dc%d' % (exp_dir, fi, ci)
                it_out_dir = '%s/%s' % (it_dir, gtex_out_dir)

                for sed_stat in sed_stats:
                    coef_out_dir = f'{it_out_dir}/coef-{sed_stat}'

                    if not os.path.isfile('%s/metrics.tsv' % coef_out_dir):
                        cmd_coef = f'{cmd_base} -o {coef_out_dir} -s {sed_stat} {it_out_dir}'
                        j = slurm.Job(cmd_coef, 'coef',
                                    f'{coef_out_dir}.out', f'{coef_out_dir}.err',
                                    queue='standard', cpu=2,
                                    mem=30000, time='12:0:0')
                        jobs.append(j)

        # ensemble
        it_out_dir = f'{exp_dir}/ensemble/{gtex_out_dir}'
        for sed_stat in sed_stats:
            coef_out_dir = f'{it_out_dir}/coef-{sed_stat}'

            if not os.path.isfile('%s/metrics.tsv' % coef_out_dir):
                cmd_coef = f'{cmd_base} -o {coef_out_dir} -s {sed_stat} {it_out_dir}'
                j = slurm.Job(cmd_coef, 'coef',
                            f'{coef_out_dir}.out', f'{coef_out_dir}.err',
                            queue='standard', cpu=2,
                            mem=30000, time='12:0:0')
                jobs.append(j)

        slurm.multi_run(jobs, verbose=True)


def collect_scores(out_dir: str, num_jobs: int, h5f_name: str='sad.h5'):
    """Collect parallel SAD jobs' output into one HDF5.

    Args:
        out_dir (str): Output directory.
        num_jobs (int): Number of jobs to combine results from.
    """
    # count variants
    num_variants = 0
    num_rows = 0
    for pi in range(num_jobs):
        # open job
        job_h5_file = '%s/job%d/%s' % (out_dir, pi, h5f_name)
        job_h5_open = h5py.File(job_h5_file, 'r')
        num_variants += len(job_h5_open['snp'])
        num_rows += len(job_h5_open['si'])
        job_h5_open.close()

    # initialize final h5
    final_h5_file = '%s/%s' % (out_dir, h5f_name)
    final_h5_open = h5py.File(final_h5_file, 'w')

    # SNP stats
    snp_stats = {}

    job0_h5_file = '%s/job0/%s' % (out_dir, h5f_name)
    job0_h5_open = h5py.File(job0_h5_file, 'r')
    for key in job0_h5_open.keys():
        if key in ['target_ids', 'target_labels']:
            # copy
            final_h5_open.create_dataset(key, data=job0_h5_open[key])

        elif key in ['snp', 'chr', 'pos', 'ref_allele', 'alt_allele', 'gene']:
            snp_stats[key] = []

        elif job0_h5_open[key].ndim == 1:
            final_h5_open.create_dataset(key, shape=(num_rows,), dtype=job0_h5_open[key].dtype)

        else:
            num_targets = job0_h5_open[key].shape[1]
            final_h5_open.create_dataset(key, shape=(num_rows, num_targets), dtype=job0_h5_open[key].dtype)

    job0_h5_open.close()

    # set values
    vgi = 0
    vi = 0
    for pi in range(num_jobs):
        # open job
        job_h5_file = '%s/job%d/%s' % (out_dir, pi, h5f_name)
        with h5py.File(job_h5_file, 'r') as job_h5_open:
            job_snps = len(job_h5_open['snp'])
            job_rows = job_h5_open['si'].shape[0]

            # append to final
            for key in job_h5_open.keys():
                try:
                    if key in ['target_ids', 'target_labels']:
                        # once is enough
                        pass

                    elif key in ['snp', 'chr', 'pos', 'ref_allele', 'alt_allele', 'gene']:
                        snp_stats[key] += list(job_h5_open[key])

                    elif key == 'si':
                        # re-index SNPs
                        final_h5_open[key][vgi:vgi+job_rows] = job_h5_open[key][:] + vi

                    else:
                        final_h5_open[key][vgi:vgi+job_rows] = job_h5_open[key]

                except TypeError as e:
                    print(e)
                    print(f'{job_h5_file} {key} has the wrong shape. Remove this file and rerun')
                    exit()

        vgi += job_rows
        vi += job_snps

    # create final SNP stat datasets
    for key in snp_stats:
        if key == 'pos':
            final_h5_open.create_dataset(key,
                data=np.array(snp_stats[key]))
        else:
            final_h5_open.create_dataset(key,
                data=np.array(snp_stats[key], dtype='S'))

    final_h5_open.close()


def ensemble_h5(ensemble_h5_file: str, scores_files: list, sed_stats: list):
    """Ensemble scores from multiple files into a single file.
    
    Args:
        ensemble_h5_file (str): ensemble score HDF5.
        scores_files ([str]): list of replicate score HDFs.
        sed_stats ([str]): SED stats to average over folds.
    """
    # open ensemble
    ensemble_h5 = h5py.File(ensemble_h5_file, 'w')

    # transfer non-SED keys
    sed_shapes = {}
    scores0_h5 = h5py.File(scores_files[0], 'r')
    for key in scores0_h5.keys():
        if key not in sed_stats:
            ensemble_h5.create_dataset(key, data=scores0_h5[key])
        else:
            sed_shapes[key] = scores0_h5[key].shape
    scores0_h5.close()

    # average stats
    num_folds = len(scores_files)
    for sed_stat in sed_stats:
        # initialize ensemble array
        sed_values = np.zeros(shape=sed_shapes[sed_stat], dtype='float32')

        # read and add folds
        for scores_file in scores_files:
            with h5py.File(scores_file, 'r') as scores_h5:
                sed_values += scores_h5[sed_stat][:].astype('float32')
        
        # normalize and downcast
        sed_values /= num_folds
        sed_values = sed_values.astype('float16')

        # save
        ensemble_h5.create_dataset(sed_stat, data=sed_values)

    ensemble_h5.close()


def split_scores(it_out_dir: str, posneg: str, vcf_dir: str, sed_stats):
    """Split merged VCF predictions in HDF5 into tissue-specific
         predictions in HDF5.
         
         Args:
             it_out_dir (str): output directory for iteration.
             posneg (str): 'pos' or 'neg'.
             vcf_dir (str): directory containing tissue-specific VCFs.
             sed_stats ([str]]): list of SED stats.
    """
    merge_h5_file = '%s/merge_%s/sed.h5' % (it_out_dir, posneg)
    merge_h5 = h5py.File(merge_h5_file, 'r')

    # read merged data
    merge_si = merge_h5['si'][:]
    merge_snps = [snp.decode('UTF-8') for snp in merge_h5['snp']]
    merge_gene = [gene.decode('UTF-8') for gene in merge_h5['gene']]
    merge_scores = {}
    for ss in sed_stats:
        merge_scores[ss] = merge_h5[ss][:]

    # hash snps to row indexes
    snp_ri = {}
    for ri, si in enumerate(merge_si):
        snp_ri.setdefault(merge_snps[si],[]).append(ri)

    # for each tissue VCF
    vcf_glob = '%s/*_%s.vcf' % (vcf_dir, posneg)
    for tissue_vcf_file in glob.glob(vcf_glob):
        tissue_label = tissue_vcf_file.split('/')[-1]
        tissue_label = tissue_label.replace('_pos.vcf','')
        tissue_label = tissue_label.replace('_neg.vcf','')

        # initialize HDF5 arrays
        sed_snp = []
        sed_chr = []
        sed_pos = []
        sed_ref = []
        sed_alt = []
        sed_gene = []
        sed_snpi = []
        sed_scores = {}
        for ss in sed_stats:
            sed_scores[ss] = []

        # fill HDF5 arrays with ordered SNPs
        si = 0
        for line in open(tissue_vcf_file):
            if not line.startswith('#'):
                a = line.split()
                chrm, pos, snp, ref, alt = a[:5]

                # SNPs w/o genes disappear
                if snp in snp_ri:
                    sed_snp.append(snp)
                    sed_chr.append(chrm)
                    sed_pos.append(int(pos))
                    sed_ref.append(ref)
                    sed_alt.append(alt)

                    for ri in snp_ri[snp]:
                        sed_snpi.append(si)
                        sed_gene.append(merge_gene[ri])
                        for ss in sed_stats:
                            sed_scores[ss].append(merge_scores[ss][ri])

                    si += 1

        # write tissue HDF5
        tissue_dir = '%s/%s_%s' % (it_out_dir, tissue_label, posneg)
        os.makedirs(tissue_dir, exist_ok=True)
        with h5py.File('%s/sed.h5' % tissue_dir, 'w') as tissue_h5:
            # write SNPs
            tissue_h5.create_dataset('snp',
                data=np.array(sed_snp, 'S'))

            # write chr
            tissue_h5.create_dataset('chr',
                data=np.array(sed_chr, 'S'))

            # write SNP pos
            tissue_h5.create_dataset('pos',
                data=np.array(sed_pos, dtype='uint32'))

            # write ref allele
            tissue_h5.create_dataset('ref_allele',
                data=np.array(sed_ref, dtype='S'))

            # write alt allele
            tissue_h5.create_dataset('alt_allele',
                data=np.array(sed_alt, dtype='S'))

            # write SNP i
            tissue_h5.create_dataset('si',
                data=np.array(sed_snpi))

            # write gene
            tissue_h5.create_dataset('gene',
                data=np.array(sed_gene, 'S'))

            # write targets
            tissue_h5.create_dataset('target_ids', data=merge_h5['target_ids'])
            tissue_h5.create_dataset('target_labels', data=merge_h5['target_labels'])

            # write sed stats
            for ss in sed_stats:
                tissue_h5.create_dataset(ss,
                    data=np.array(sed_scores[ss], dtype='float16'))

    merge_h5.close()

def nonzero_h5(h5_file: str, stat_keys):
    """Verify the HDF5 exists, and there are nonzero values
        for each stat key given.

    Args:
        h5_file (str): HDF5 file name.
        stat_keys ([str]): List of SNP stat keys.
    """
    if os.path.isfile(h5_file):
        try:
            with h5py.File(h5_file, 'r') as h5_open:
                for sk in stat_keys:
                    sad = h5_open[sk][:]
                    if (sad != 0).sum() > 0:
                        return True
                return False
        except:
            return False
    else:
        return False

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
