#!/usr/bin/env python
# Copyright 2021 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from optparse import OptionParser
import gc
import json
import pdb
import os
import time
import sys

import h5py
#from intervaltree import IntervalTree
import numpy as np
import pandas as pd
import pyranges as pr
from scipy.stats import pearsonr
from sklearn.metrics import explained_variance_score
import tensorflow as tf
#from tqdm import tqdm

from basenji import bed
from basenji import dataset
from basenji import seqnn
from basenji import trainer
#import pygene
#from qnorm import quantile_normalize

'''
borzoi_test_tss_gencode.py

Measure accuracy at TSS-level.
'''

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <model_file> <data_dir> <exons_gff>'
    parser = OptionParser(usage)
    parser.add_option(
        '--head',
        dest='head_i',
        default=0,
        type='int',
        help='Parameters head [Default: %default]',
    )
    parser.add_option(
        '-o',
        dest='out_dir',
        default='teste_out',
        help='Output directory for predictions [Default: %default]',
    )
    parser.add_option(
        '--rc',
        dest='rc',
        default=False,
        action='store_true',
        help='Average the fwd and rc predictions [Default: %default]',
    )
    parser.add_option(
        '--shifts',
        dest='shifts',
        default='0',
        help='Ensemble prediction shifts [Default: %default]',
    )
    parser.add_option(
        '--windowcov',
        dest='windowcov',
        default=4,
        type='int',
        help='Coverage bin window size [Default: %default]',
    )
    parser.add_option(
        '--maxcov',
        dest='maxcov',
        default=False,
        action='store_true',
        help='Store max instead of avg bin value in local window [Default: %default]',
    )
    parser.add_option(
        '-t',
        dest='targets_file',
        default=None,
        type='str',
        help='File specifying target indexes and labels in table format',
    )
    parser.add_option(
        '--split',
        dest='split_label',
        default='test',
        help='Dataset split label for eg TFR pattern [Default: %default]',
    )
    parser.add_option(
        '--tfr',
        dest='tfr_pattern',
        default=None,
        help='TFR pattern string appended to data_dir/tfrecords for subsetting [Default: %default]',
    )
    (options, args) = parser.parse_args()

    if len(args) != 4:
        parser.error('Must provide parameters, model, data directory, and TSS annotation')
    else:
        params_file = args[0]
        model_file = args[1]
        data_dir = args[2]
        tss_file = args[3]
    
    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    # parse shifts to integers
    options.shifts = [int(shift) for shift in options.shifts.split(',')]

    #######################################################
    # inputs
    
    # read targets
    if options.targets_file is None:
        options.targets_file = '%s/targets.txt' % data_dir
    targets_df = pd.read_csv(options.targets_file, index_col=0, sep='\t')

    # attach strand
    targets_strand = []
    for ti, identifier in enumerate(targets_df.identifier):
        if targets_df.index[ti] == targets_df.strand_pair.iloc[ti]:
            targets_strand.append('.')
        else:
            targets_strand.append(identifier[-1])
    targets_df['strand'] = targets_strand

    # collapse stranded
    strand_mask = (targets_df.strand != '-')
    targets_strand_df = targets_df[strand_mask]

    # count targets
    num_targets = targets_df.shape[0]
    num_targets_strand = targets_strand_df.shape[0]

    # save sqrt'd tracks
    sqrt_mask = np.array([ss.find('sqrt') != -1 for ss in targets_strand_df.sum_stat])

    # read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
    params_model = params['model']
    params_train = params['train']

    # set strand pairs
    if 'strand_pair' in targets_df.columns:
        orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
        targets_strand_pair = np.array([orig_new_index[ti] for ti in targets_df.strand_pair])
        params_model['strand_pair'] = [targets_strand_pair]
    
    # construct eval data
    eval_data = dataset.SeqDataset(data_dir,
        split_label=options.split_label,
        batch_size=params_train['batch_size'],
        mode='eval',
        tfr_pattern=options.tfr_pattern)

    # initialize model
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file, options.head_i)
    seqnn_model.build_slice(targets_df.index)
    seqnn_model.build_ensemble(options.rc, options.shifts)

    #######################################################
    # sequence intervals

    # read data parameters
    with open('%s/statistics.json'%data_dir) as data_open:
        data_stats = json.load(data_open)
        crop_bp = data_stats['crop_bp']
        pool_width = data_stats['pool_width']

    # read sequence positions
    seqs_df = pd.read_csv('%s/sequences.bed'%data_dir, sep='\t',
        names=['Chromosome','Start','End','Name'])
    seqs_df = seqs_df[seqs_df.Name == options.split_label]
    seqs_pr = pr.PyRanges(seqs_df)

    #######################################################
    # make TSS BED (GENCODE)
    
    tss_df = pd.read_csv(tss_file, sep='\t', names=['Chromosome', 'Start', 'End', 'tss_id', 'feat1', 'tss_strand'])
    
    tss_pr = pr.PyRanges(tss_df)

    #######################################################
    # intersect TSS sites w/ preds, targets
    
    # intersect seqs, TSS sites
    seqs_tss_pr = seqs_pr.join(tss_pr)
    
    eprint("len(seqs_tss_pr.df) = " + str(len(seqs_tss_pr.df)))
    print("len(seqs_tss_pr.df) = " + str(len(seqs_tss_pr.df)))
    
    # hash preds/targets by tss_id
    tss_preds_dict = {}
    tss_targets_dict = {}

    si = 0
    for x, y in eval_data.dataset:
        # predict only if gene overlaps
        yh = None
        y = y.numpy()[...,targets_df.index]

        t0 = time.time()
        eprint('Sequence %d...' % si)
        print('Sequence %d...' % si, end='')
        for bsi in range(x.shape[0]):
            seq = seqs_df.iloc[si+bsi]

            cseqs_tss_df = seqs_tss_pr[seq.Chromosome].df
            if cseqs_tss_df.shape[0] == 0:
                # empty. no tss sites on this chromosome
                seq_tss_df = cseqs_tss_df
            else:
                seq_tss_df = cseqs_tss_df[cseqs_tss_df.Start == seq.Start]

            for _, seq_tss in seq_tss_df.iterrows():
                tss_id = seq_tss.tss_id
                tss_start = seq_tss.Start_b
                tss_end = seq_tss.End_b
                seq_start = seq_tss.Start
                tss_strand = seq_tss.tss_strand

                # clip boundaries
                tss_seq_start = max(0, tss_start - seq_start)
                tss_seq_end = max(0, tss_end - seq_start)

                # requires >50% overlap
                
                bin_start = None
                bin_end = None
                if tss_strand == '+' :
                        bin_start = int(np.round(tss_seq_end / pool_width))
                        bin_end = bin_start + options.windowcov
                else :
                        bin_end = int(np.round(tss_seq_start / pool_width)) + 1
                        bin_start = bin_end - options.windowcov

                # predict
                if yh is None:
                    yh = seqnn_model(x)

                # slice gene region
                yhb = yh[bsi,bin_start:bin_end].astype('float16')
                yb = y[bsi,bin_start:bin_end].astype('float16')

                if len(yb) > 0:    
                    tss_preds_dict.setdefault(tss_id,[]).append(yhb)
                    tss_targets_dict.setdefault(tss_id,[]).append(yb)
                else:
                    eprint("(Warning: len(yb) <= 0)")
        
        # advance sequence table index
        si += x.shape[0]
        eprint('DONE in %ds.' % (time.time()-t0))
        print('DONE in %ds.' % (time.time()-t0))
        
        eprint("len(tss_preds_dict) = " + str(len(tss_preds_dict)))
        
        if si % 128 == 0:
            gc.collect()


    #######################################################
    # aggregate TSS bin values into arrays

    tss_targets = []
    tss_preds = []
    tss_ids = np.array(sorted(tss_targets_dict.keys()))

    for tss_id in tss_ids:
        tss_preds_gi = np.concatenate(tss_preds_dict[tss_id], axis=0).astype('float32')
        tss_targets_gi = np.concatenate(tss_targets_dict[tss_id], axis=0).astype('float32')

        # undo scale
        tss_preds_gi /= np.expand_dims(targets_strand_df.scale, axis=0)
        tss_targets_gi /= np.expand_dims(targets_strand_df.scale, axis=0)

        # undo sqrt
        tss_preds_gi[:,sqrt_mask] = tss_preds_gi[:,sqrt_mask]**(4/3)
        tss_targets_gi[:,sqrt_mask] = tss_targets_gi[:,sqrt_mask]**(4/3)

        # mean (or max) coverage
        tss_preds_gi = tss_preds_gi.max(axis=0) if options.maxcov else tss_preds_gi.mean(axis=0)
        tss_targets_gi = tss_targets_gi.max(axis=0) if options.maxcov else tss_targets_gi.mean(axis=0)

        tss_preds.append(tss_preds_gi)
        tss_targets.append(tss_targets_gi)

    tss_targets = np.array(tss_targets)
    tss_preds = np.array(tss_preds)

    # save numpy arrays with values
    np.save('%s/tss_targets_gencode.npy' % options.out_dir, tss_targets)
    np.save('%s/tss_preds_gencode.npy' % options.out_dir, tss_preds)

    # save values
    tss_targets_df = pd.DataFrame(tss_targets, index=tss_ids)
    tss_targets_df.to_csv('%s/tss_targets_gencode.tsv.gz' % options.out_dir, sep='\t')
    tss_preds_df = pd.DataFrame(tss_preds, index=tss_ids)
    tss_preds_df.to_csv('%s/tss_preds_gencode.tsv.gz' % options.out_dir, sep='\t')

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
