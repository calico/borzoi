#!/usr/bin/env python
# Copyright 2017 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from __future__ import print_function

from optparse import OptionParser

import gc
import json
import os
import pdb
import pickle
from queue import Queue
import random
import sys
from threading import Thread
import time

import h5py
import numpy as np
import pandas as pd
import pysam
import pygene
import tensorflow as tf

from baskerville import dna as dna_io
from baskerville import gene as bgene
from baskerville import seqnn
from baskerville.dataset import targets_prep_strand

from scipy.ndimage import gaussian_filter1d

'''
borzoi_satg_gene_crispr_ism_shuffle.py

Perform a windowed shuffle analysis for genes specified in a GTF file, targeting regions specified in a separate csv.
'''


################################################################################
# main
# ###############################################################################
def main():
    usage = 'usage: %prog [options] <params> <model> <gene_gtf>'
    parser = OptionParser(usage)
    parser.add_option(
        "--fa",
        dest="genome_fasta",
        default="%s/assembly/ucsc/hg38.fa" % os.environ.get('BORZOI_HG38', 'hg38'),
        help="Genome FASTA for sequences [Default: %default]",
    )
    parser.add_option(
        "-o",
        dest="out_dir",
        default="satg_out",
        help="Output directory [Default: %default]",
    )
    parser.add_option(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Ensemble forward and reverse complement predictions [Default: %default]",
    )
    parser.add_option(
        "-f",
        dest="folds",
        default="0",
        type="str",
        help="Model folds to use in ensemble (comma-separated list) [Default: %default]",
    )
    parser.add_option(
        '-c',
        dest='crosses',
        default="0",
        type="str",
        help='Model crosses (replicates) to use in ensemble (comma-separated list) [Default:%default]',
    )
    parser.add_option(
        "--head",
        dest="head_i",
        default=0,
        type="int",
        help="Model head index [Default: %default]",
    )
    parser.add_option(
        "--shifts",
        dest="shifts",
        default="0",
        type="str",
        help="Ensemble prediction shifts [Default: %default]",
    )
    parser.add_option(
        "--span",
        dest="span",
        default=False,
        action="store_true",
        help="Aggregate entire gene span [Default: %default]",
    )
    parser.add_option(
        "--clip_soft",
        dest="clip_soft",
        default=None,
        type="float",
        help="Model clip_soft setting [Default: %default]",
    )
    parser.add_option(
        "--track_scale",
        dest="track_scale",
        default=0.02,
        type="float",
        help="Target transform scale [Default: %default]",
    )
    parser.add_option(
        "--track_transform",
        dest="track_transform",
        default=0.75,
        type="float",
        help="Target transform exponent [Default: %default]",
    )
    parser.add_option(
        "--untransform_old",
        dest="untransform_old",
        default=False,
        action="store_true",
        help="Run gradients with old version of inverse transforms [Default: %default]",
    )
    parser.add_option(
        "--no_untransform",
        dest="no_untransform",
        default=False,
        action="store_true",
        help="Run gradients with no inverse transforms [Default: %default]",
    )
    parser.add_option(
        '--pseudo',
        dest='pseudo_count',
        default=0,
        type='float',
        help='Pseudo count (untransformed count space) [Default: %default]',
    )
    parser.add_option(
        '--aggregate_tracks',
        dest='aggregate_tracks',
        default=None,
        type='int',
        help='Aggregate groups of tracks [Default: %default]',
    )
    parser.add_option(
        '-t',
        dest='targets_file',
        default=None,
        type='str',
        help='File specifying target indexes and labels in table format',
    )
    parser.add_option(
        '--ism_size',
        dest='ism_size',
        default=192,
        type='int',
        help='Length of sequence window to run ISM across.',
    )
    parser.add_option(
        '--crispr_file',
        dest='crispr_file',
        default=None,
        type='str',
        help='Tsv-file with gene/crispr metadata',
    )
    parser.add_option(
        '--window_size',
        dest='window_size',
        default=5,
        type='int',
        help='ISM shuffle window size [Default: %default]',
    )
    parser.add_option(
        '--n_samples',
        dest='n_samples',
        default=8,
        type='int',
        help='ISM shuffle samples per position [Default: %default]',
    )
    parser.add_option(
        '--mononuc_shuffle',
        dest='mononuc_shuffle',
        default=False,
        action="store_true",
        help='Mono-nucleotide shuffle [Default: %default]',
    )
    parser.add_option(
        '--dinuc_shuffle',
        dest='dinuc_shuffle',
        default=False,
        action="store_true",
        help='Di-nucleotide shuffle [Default: %default]',
    )
    (options, args) = parser.parse_args()

    if len(args) == 3:
        # single worker
        params_file = args[0]
        model_folder = args[1]
        genes_gtf_file = args[2]
    else:
        parser.error('Must provide parameter file, model folder and BED file')

    if not os.path.isdir(options.out_dir):
        os.makedirs(options.out_dir, exist_ok=True)

    options.folds = [int(fold) for fold in options.folds.split(',')]
    options.crosses = [int(cross) for cross in options.crosses.split(",")]
    options.shifts = [int(shift) for shift in options.shifts.split(',')]

    #################################################################
    # read parameters and targets

    # read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
    params_model = params['model']
    params_train = params['train']
    seq_len = params_model['seq_length']

    if options.targets_file is None:
        parser.error('Must provide targets table to properly handle strands.')
    else:
        targets_df = pd.read_csv(options.targets_file, sep='\t', index_col=0)

    # prep strand
    orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
    targets_strand_pair = np.array([orig_new_index[ti] for ti in targets_df.strand_pair])
    targets_strand_df = targets_prep_strand(targets_df)
    num_targets = len(targets_strand_df)

    # specify relative target indices
    targets_df['row_index'] = np.arange(len(targets_df), dtype='int32')

    #################################################################
    # load first model fold to get parameters

    seqnn_model = seqnn.SeqNN(params_model)
    
    model_path = model_folder + "/f" + str(options.folds[0]) + "c0/train/model" + str(options.head_i) + "_best.h5"
    if not os.path.isfile(model_path) :
        model_path = model_folder + "/f" + str(options.folds[0]) + "c0/train/model_best.h5"
    
    seqnn_model.restore(
        model_path,
        options.head_i
    )
    seqnn_model.build_slice(targets_df.index, False)
    # seqnn_model.build_ensemble(options.rc, options.shifts)

    model_stride = seqnn_model.model_strides[0]
    model_crop = seqnn_model.target_crops[0]
    target_length = seqnn_model.target_lengths[0]

    #################################################################
    # read genes

    # parse GTF
    transcriptome = bgene.Transcriptome(genes_gtf_file)

    # order valid genes
    genome_open = pysam.Fastafile(options.genome_fasta)
    gene_list = sorted(transcriptome.genes.keys())
    num_genes = len(gene_list)
    
    print("num_genes = " + str(num_genes))
    
    #################################################################
    # gene/crispr metadata
    
    #Load gene/crispr dataframe
    crispr_df = pd.read_csv(options.crispr_file, sep='\t')

    print("len(crispr_df) = " + str(len(crispr_df)))
    
    #Map gene_ids to names
    gene_id_to_name_dict = {}
    gene_name_to_id_dict = {}
    with open(genes_gtf_file) as gtf_f :
        for line in gtf_f.readlines():
            a = line.split('\t')
            kv = pygene.gtf_kv(a[8])
            
            gene_id_to_name_dict[kv['gene_id']] = kv['gene_name']
            gene_name_to_id_dict[kv['gene_name']] = kv['gene_id']
    
    #Collect pseudo count
    pseudo_count = options.pseudo_count
    print("pseudo_count = " + str(round(pseudo_count, 2)))

    #################################################################
    # setup output

    min_start = -model_stride*model_crop

    # choose gene sequences
    genes_chr = []
    genes_start = []
    genes_end = []
    genes_strand = []
    for gene_id in gene_list:
        gene = transcriptome.genes[gene_id]
        genes_chr.append(gene.chrom)
        genes_strand.append(gene.strand)

        gene_midpoint = gene.midpoint()
        gene_start = max(min_start, gene_midpoint - seq_len//2)
        gene_end = gene_start + seq_len
        genes_start.append(gene_start)
        genes_end.append(gene_end)
    
    #################################################################
    # calculate ism start and end positions per gene
    
    genes_ism_regions = []
    for gi in range(len(gene_list)) :
        
        gene_id = gene_list[gi]
        gene_name = gene_id_to_name_dict[gene_id]
        
        crispr_df_gene = crispr_df.loc[crispr_df['gene'] == gene_name].copy().reset_index(drop=True)
        
        gene_start = genes_start[gi]
        
        ism_regions = []
        for _, row in crispr_df_gene.iterrows() :
            site_mid = (row['start'] + row['end']) // 2
            rel_pos = site_mid - gene_start
                
            ism_start = rel_pos - options.ism_size // 2
            ism_end = rel_pos + options.ism_size // 2 + options.ism_size % 2
            
            if ism_start - options.window_size // 2 >= 0 and ism_end + options.window_size // 2 + 1 < seq_len :
                ism_regions.append([ism_start, ism_end])
        
        genes_ism_regions.append(ism_regions)

    #################################################################
    # predict ISM scores, write output
    
    print("clip_soft = " + str(options.clip_soft))
    
    print("n genes = " + str(len(genes_chr)))
    
    # loop over folds
    for fold_ix in options.folds :
        for cross_ix in options.crosses:
            
            print("-- fold = f" + str(fold_ix) + "c" + str(cross_ix) + " --")

            # (re-)initialize HDF5
            scores_h5_file = '%s/ism_f%dc%d.h5' % (options.out_dir, fold_ix, cross_ix)
            if os.path.isfile(scores_h5_file):
                os.remove(scores_h5_file)
            scores_h5 = h5py.File(scores_h5_file, 'w')
            scores_h5.create_dataset('seqs', dtype='bool',
                shape=(num_genes, seq_len, 4))
            scores_h5.create_dataset('isms', dtype='float16',
                shape=(num_genes, seq_len, 4, num_targets // (options.aggregate_tracks if options.aggregate_tracks is not None else 1)))
            scores_h5.create_dataset('gene', data=np.array(gene_list, dtype='S'))
            scores_h5.create_dataset('chr', data=np.array(genes_chr, dtype='S'))
            scores_h5.create_dataset('start', data=np.array(genes_start))
            scores_h5.create_dataset('end', data=np.array(genes_end))
            scores_h5.create_dataset('strand', data=np.array(genes_strand, dtype='S'))

            # load model fold
            seqnn_model = seqnn.SeqNN(params_model)

            model_path = model_folder + "/f" + str(fold_ix) + "c" + str(cross_ix) + "/train/model" + str(options.head_i) + "_best.h5"
            if not os.path.isfile(model_path) :
                model_path = model_folder + "/f" + str(fold_ix) + "c" + str(cross_ix) + "/train/model_best.h5"

            seqnn_model.restore(
                model_path,
                options.head_i
            )
            seqnn_model.build_slice(targets_df.index, False)

            for shift in options.shifts :
                print('Processing shift %d' % shift, flush=True)

                for rev_comp in ([False, True] if options.rc else [False]) :

                    if options.rc :
                        print('Fwd/rev = %s' % ('fwd' if not rev_comp else 'rev'), flush=True)

                    for gi, gene_id in enumerate(gene_list):

                        if gi % 5 == 0 :
                            print('Processing %d, %s' % (gi, gene_id), flush=True)

                        gene = transcriptome.genes[gene_id]

                        # make sequence
                        seq_1hot = make_seq_1hot(genome_open, genes_chr[gi], genes_start[gi], genes_end[gi], seq_len)
                        seq_1hot = dna_io.hot1_augment(seq_1hot, shift=shift)

                        # determine output sequence start
                        seq_out_start = genes_start[gi] + model_stride*model_crop
                        seq_out_len = model_stride*target_length

                        # determine output positions
                        gene_slice = gene.output_slice(seq_out_start, seq_out_len, model_stride, options.span)

                        # get ism window regions
                        gene_ism_regions = genes_ism_regions[gi]

                        if rev_comp:
                            seq_1hot = dna_io.hot1_rc(seq_1hot)
                            gene_slice = target_length - gene_slice - 1

                            gene_ism_regions = []
                            for [genes_ism_start_orig, gene_ism_end_orig] in genes_ism_regions[gi] :
                                gene_ism_start = seq_len - gene_ism_end_orig - 1
                                gene_ism_end = seq_len - genes_ism_start_orig - 1

                                gene_ism_regions.append([gene_ism_start, gene_ism_end])

                        # slice relevant strand targets
                        if genes_strand[gi] == '+':
                            gene_strand_mask = (targets_df.strand != '-') if not rev_comp else (targets_df.strand != '+')
                        else:
                            gene_strand_mask = (targets_df.strand != '+') if not rev_comp else (targets_df.strand != '-')

                        gene_target = np.array(targets_df.index[gene_strand_mask].values)

                        # broadcast to singleton batch
                        seq_1hot = seq_1hot[None, ...]
                        gene_slice = gene_slice[None, ...]
                        gene_target = gene_target[None, ...]

                        # ism computation
                        ism = get_ism_shuffle(
                                seqnn_model,
                                seq_1hot,
                                gene_ism_regions,
                                head_i=0,
                                target_slice=gene_target,
                                pos_slice=gene_slice,
                                track_scale=options.track_scale,
                                track_transform=options.track_transform,
                                clip_soft=options.clip_soft,
                                pseudo_count=pseudo_count,
                                untransform_old=options.untransform_old,
                                no_untransform=options.no_untransform,
                                aggregate_tracks=options.aggregate_tracks,
                                use_mean=False,
                                use_ratio=False,
                                use_logodds=False,
                                window_size=options.window_size,
                                n_samples=options.n_samples,
                                mononuc_shuffle=options.mononuc_shuffle,
                                dinuc_shuffle=options.dinuc_shuffle,
                        )

                        # undo augmentations and save ism
                        ism = unaugment_grads(ism, fwdrc=(not rev_comp), shift=shift)

                        # write to HDF5
                        scores_h5['isms'][gi] += ism[:, ...]

                        # collect garbage
                        gc.collect()

            # save sequences and normalize isms by total size of ensemble
            for gi, gene_id in enumerate(gene_list):

                # re-make original sequence
                seq_1hot = make_seq_1hot(genome_open, genes_chr[gi], genes_start[gi], genes_end[gi], seq_len)

                # write to HDF5
                scores_h5['seqs'][gi] = seq_1hot[:, ...]
                scores_h5['isms'][gi] /= float((len(options.shifts) * (2 if options.rc else 1)))

            # collect garbage
            gc.collect()

    # close files
    genome_open.close()
    scores_h5.close()


def unaugment_grads(grads, fwdrc=False, shift=0):
    """ Undo sequence augmentation."""
    # reverse complement
    if not fwdrc:
        # reverse
        grads = grads[::-1, :, :]

        # swap A and T
        grads[:, [0, 3], :] = grads[:, [3, 0], :]

        # swap C and G
        grads[:, [1, 2], :] = grads[:, [2, 1], :]

    # undo shift
    if shift < 0:
        # shift sequence right
        grads[-shift:, :, :] = grads[:shift, :, :]

        # fill in left unknowns
        grads[:-shift, :, :] = 0

    elif shift > 0:
        # shift sequence left
        grads[:-shift, :, :] = grads[shift:, :, :]

        # fill in right unknowns
        grads[-shift:, :, :] = 0

    return grads


def make_seq_1hot(genome_open, chrm, start, end, seq_len):
    if start < 0:
        seq_dna = 'N'*(-start) + genome_open.fetch(chrm, 0, end)
    else:
        seq_dna = genome_open.fetch(chrm, start, end)
        
    # extend to full length
    if len(seq_dna) < seq_len:
        seq_dna += 'N'*(seq_len-len(seq_dna))

    seq_1hot = dna_io.dna_1hot(seq_dna)
    return seq_1hot


# tf code for computing ISM scores on GPU
@tf.function
def _score_func(model, seq_1hot, target_slice, pos_slice, pos_mask=None, pos_slice_denom=None, pos_mask_denom=True, track_scale=1., track_transform=1., clip_soft=None, pseudo_count=0., untransform_old=False, no_untransform=False, aggregate_tracks=None, use_mean=False, use_ratio=False, use_logodds=False) :
            
    # predict
    preds = tf.gather(model(seq_1hot, training=False), target_slice, axis=-1, batch_dims=1)

    if not no_untransform:
        if untransform_old:
            # undo scale
            preds = preds / track_scale

            # undo soft_clip
            if clip_soft is not None:
                preds = tf.where(
                    preds > clip_soft, (preds - clip_soft) ** 2 + clip_soft, preds
                )

            # undo sqrt
            preds = preds ** (1. / track_transform)
        else:
            # undo clip_soft
            if clip_soft is not None:
                preds = tf.where(
                    preds > clip_soft, (preds - clip_soft + 1) ** 2 + clip_soft - 1, preds
                )

            # undo sqrt
            preds = -1 + (preds + 1) ** (1. / track_transform)

            # scale
            preds = preds / track_scale

    if aggregate_tracks is not None :
        preds = tf.reduce_mean(tf.reshape(preds, (preds.shape[0], preds.shape[1], preds.shape[2] // aggregate_tracks, aggregate_tracks)), axis=-1)

    # slice specified positions
    preds_slice = tf.gather(preds, pos_slice, axis=1, batch_dims=1)
    if pos_mask is not None :
        preds_slice = preds_slice * pos_mask

    # slice denominator positions
    if use_ratio and pos_slice_denom is not None:
        preds_slice_denom = tf.gather(preds, pos_slice_denom, axis=1, batch_dims=1)
        if pos_mask_denom is not None :
            preds_slice_denom = preds_slice_denom * pos_mask_denom

    # aggregate over positions
    if not use_mean :
        preds_agg = tf.reduce_sum(preds_slice, axis=1)
        if use_ratio and pos_slice_denom is not None:
            preds_agg_denom = tf.reduce_sum(preds_slice_denom, axis=1)
    else :
        if pos_mask is not None :
            preds_agg = tf.reduce_sum(preds_slice, axis=1) / tf.reduce_sum(pos_mask, axis=1)
        else :
            preds_agg = tf.reduce_mean(preds_slice, axis=1)

        if use_ratio and pos_slice_denom is not None:
            if pos_mask_denom is not None :
                preds_agg_denom = tf.reduce_sum(preds_slice_denom, axis=1) / tf.reduce_sum(pos_mask_denom, axis=1)
            else :
                preds_agg_denom = tf.reduce_mean(preds_slice_denom, axis=1)

    # compute final statistic
    if no_untransform :
        score_ratios = preds_agg
    elif not use_ratio :
        score_ratios = tf.math.log(preds_agg + pseudo_count + 1e-6)
    else :
        if not use_logodds :
            score_ratios = tf.math.log((preds_agg + pseudo_count) / (preds_agg_denom + pseudo_count) + 1e-6)
        else :
            score_ratios = tf.math.log(((preds_agg + pseudo_count) / (preds_agg_denom + pseudo_count)) / (1. - ((preds_agg + pseudo_count) / (preds_agg_denom + pseudo_count))) + 1e-6)

    return score_ratios


def get_ism_shuffle(seqnn_model, seq_1hot_wt, ism_regions, head_i=None, target_slice=None, pos_slice=None, pos_mask=None, pos_slice_denom=None, pos_mask_denom=None, track_scale=1., track_transform=1., clip_soft=None, pseudo_count=0., untransform_old=False, no_untransform=False, aggregate_tracks=None, use_mean=False, use_ratio=False, use_logodds=False, bases=[0, 1, 2, 3], window_size=5, n_samples=8, mononuc_shuffle=False, dinuc_shuffle=False) :
    
    # choose model
    if seqnn_model.ensemble is not None:
        model = seqnn_model.ensemble
    elif head_i is not None:
        model = seqnn_model.models[head_i]
    else:
        model = seqnn_model.model
    
    # verify tensor shape(s)
    seq_1hot_wt = seq_1hot_wt.astype('float32')
    target_slice = np.array(target_slice).astype('int32')
    pos_slice = np.array(pos_slice).astype('int32')
    
    # convert constants to tf tensors
    track_scale = tf.constant(track_scale, dtype=tf.float32)
    track_transform = tf.constant(track_transform, dtype=tf.float32)
    if clip_soft is not None :
        clip_soft = tf.constant(clip_soft, dtype=tf.float32)
    pseudo_count = tf.constant(pseudo_count, dtype=tf.float32)
    
    if pos_mask is not None :
        pos_mask = np.array(pos_mask).astype('float32')
    
    if use_ratio and pos_slice_denom is not None :
        pos_slice_denom = np.array(pos_slice_denom).astype('int32')
      
        if pos_mask_denom is not None :
            pos_mask_denom = np.array(pos_mask_denom).astype('float32')
    
    if len(seq_1hot_wt.shape) < 3:
        seq_1hot_wt = seq_1hot_wt[None, ...]
    
    if len(target_slice.shape) < 2:
        target_slice = target_slice[None, ...]
    
    if len(pos_slice.shape) < 2:
        pos_slice = pos_slice[None, ...]
    
    if pos_mask is not None and len(pos_mask.shape) < 2:
        pos_mask = pos_mask[None, ...]
    
    if use_ratio and pos_slice_denom is not None and len(pos_slice_denom.shape) < 2:
        pos_slice_denom = pos_slice_denom[None, ...]
      
        if pos_mask_denom is not None and len(pos_mask_denom.shape) < 2:
            pos_mask_denom = pos_mask_denom[None, ...]
    
    # convert to tf tensors
    seq_1hot_wt_tf = tf.convert_to_tensor(seq_1hot_wt, dtype=tf.float32)
    target_slice = tf.convert_to_tensor(target_slice, dtype=tf.int32)
    pos_slice = tf.convert_to_tensor(pos_slice, dtype=tf.int32)
    
    if pos_mask is not None :
        pos_mask = tf.convert_to_tensor(pos_mask, dtype=tf.float32)
    
    if use_ratio and pos_slice_denom is not None :
        pos_slice_denom = tf.convert_to_tensor(pos_slice_denom, dtype=tf.int32)
    
        if pos_mask_denom is not None :
            pos_mask_denom = tf.convert_to_tensor(pos_mask_denom, dtype=tf.float32)
    
    # allocate ism shuffle result tensor
    pred_shuffle = np.zeros((seq_1hot_wt.shape[1], n_samples, target_slice.shape[1] // (aggregate_tracks if aggregate_tracks is not None else 1)))
    
    # get wt pred
    score_wt = _score_func(model, seq_1hot_wt_tf, target_slice, pos_slice, pos_mask, pos_slice_denom, pos_mask_denom, track_scale, track_transform, clip_soft, pseudo_count, untransform_old, no_untransform, aggregate_tracks, use_mean, use_ratio, use_logodds).numpy()
    
    for ism_region_i, [ism_start, ism_end] in enumerate(ism_regions) :
        for j in range(ism_start, ism_end) :
            j_start = j - window_size // 2
            j_end = j + window_size // 2 + 1

            pos_index = np.arange(j_end - j_start) + j_start
            
            for sample_ix in range(n_samples):
                seq_1hot_mut = np.copy(seq_1hot_wt)
                seq_1hot_mut[0, j_start:j_end, :] = 0.
                
                if not mononuc_shuffle and not dinuc_shuffle:
                    nt_index = np.random.choice(bases, size=(j_end - j_start,)).tolist()
                    seq_1hot_mut[0, pos_index, nt_index] = 1.
                elif mononuc_shuffle:
                    shuffled_pos_index = np.copy(pos_index)
                    np.random.shuffle(shuffled_pos_index)

                    seq_1hot_mut[0, shuffled_pos_index, :] = seq_1hot_wt[0, pos_index, :]
                else:  # dinuc-shuffle
                    shuffled_pos_index = [
                        [pos_index[pos_j], pos_index[pos_j + 1]]
                        if pos_j + 1 < pos_index.shape[0] else [pos_index[pos_j]]
                        for pos_j in range(0, pos_index.shape[0], 2)
                    ]

                    shuffled_shuffle_index = np.arange(len(shuffled_pos_index), dtype="int32")
                    np.random.shuffle(shuffled_shuffle_index)

                    shuffled_pos_index_new = []
                    for pos_tuple_i in range(len(shuffled_pos_index)):
                        shuffled_pos_index_new.extend(
                            shuffled_pos_index[shuffled_shuffle_index[pos_tuple_i]]
                        )

                    shuffled_pos_index = np.array(shuffled_pos_index_new, dtype="int32")
                    seq_1hot_mut[0, shuffled_pos_index, :] = seq_1hot_wt[0, pos_index, :]
            
                # convert to tf tensor
                seq_1hot_mut_tf = tf.convert_to_tensor(seq_1hot_mut, dtype=tf.float32)

                # get mut pred
                score_mut = _score_func(model, seq_1hot_mut_tf, target_slice, pos_slice, pos_mask, pos_slice_denom, pos_mask_denom, track_scale, track_transform, clip_soft, pseudo_count, untransform_old, no_untransform, aggregate_tracks, use_mean, use_ratio, use_logodds).numpy()

                pred_shuffle[j, sample_ix, :] = score_wt - score_mut

    pred_ism = np.tile(np.mean(pred_shuffle, axis=1, keepdims=True), (1, 4, 1)) * seq_1hot_wt[0, ..., None]
    
    return pred_ism


################################################################################
# __main__
# ###############################################################################
if __name__ == '__main__':
    main()
