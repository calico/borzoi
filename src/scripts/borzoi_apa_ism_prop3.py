#!/usr/bin/env python
# Copyright 2023 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#                 https://www.apache.org/licenses/LICENSE-2.0

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
import itertools

import h5py
import numpy as np
import pandas as pd
import pysam
import tensorflow as tf

from baskerville import dna as dna_io
from baskerville import gene as bgene
from baskerville import seqnn
from baskerville.dataset import targets_prep_strand

'''
borzoi_apa_ism_prop3.py

Perform an ISM analysis for APA sites / UTRs specified in a csv file (proportion statistics).
'''

################################################################################
# main
# ###############################################################################
def main():
    usage = "usage: %prog [options] <params> <model> <gene_csv>"
    parser = OptionParser(usage)
    parser.add_option(
        "--fa",
        dest="genome_fasta",
        default="%s/assembly/ucsc/hg38.fa" % os.environ.get("BORZOI_HG38", "hg38"),
        help="Genome FASTA for sequences [Default: %default]",
    )
    parser.add_option(
        "--head",
        dest="head_i",
        default=None,
        type="int",
        help="Parameters head [Default: %default]",
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
        "--separate_rc",
        dest="separate_rc",
        default=False,
        action="store_true",
        help="Store reverse complement scores separately (do not average with forward scores) [Default: %default]",
    )
    parser.add_option(
        "-f",
        dest="folds",
        default="0",
        type="str",
        help="Model folds to use in ensemble [Default: %default]",
    )
    parser.add_option(
        "-c",
        dest="crosses",
        default="0",
        type="str",
        help="Model crosses to use in ensemble [Default: %default]",
    )
    parser.add_option(
        "--shifts",
        dest="shifts",
        default="0",
        type="str",
        help="Ensemble prediction shifts [Default: %default]",
    )
    parser.add_option(
        "--paext",
        dest="pas_ext",
        default=100,
        type="int",
        help="Extension in bp past UTR span annotation [Default: %default]",
    )
    parser.add_option(
        "--upstream_paext",
        dest="pas_ext_up",
        default=None,
        type="int",
        help="Extension in bp past UTR span annotation, upstream only [Default: %default]",
    )
    parser.add_option(
        "-t",
        dest="targets_file",
        default=None,
        type="str",
        help="File specifying target indexes and labels in table format",
    )
    parser.add_option(
        "--ism_size",
        dest="ism_size",
        default=192,
        type="int",
        help="Length of sequence window to run ISM across (centered at pA site)",
    )
    parser.add_option(
        "--splice_ism_size",
        dest="splice_ism_size",
        default=64,
        type="int",
        help="Length of sequence window to run ISM across (centered at splice site)",
    )
    parser.add_option(
        "--do_shuffle",
        dest="do_shuffle",
        default=False,
        action="store_true",
        help="Run ISM Shuffle (otherwise ISM) [Default: %default]",
    )    
    parser.add_option(
        "--window_size",
        dest="window_size",
        default=1,
        type="int",
        help="ISM shuffle window size [Default: %default]",
    )
    parser.add_option(
        "--n_samples",
        dest="n_samples",
        default=8,
        type="int",
        help="ISM shuffle samples per position [Default: %default]",
    )
    parser.add_option(
        "--mononuc_shuffle",
        dest="mononuc_shuffle",
        default=False,
        action="store_true",
        help="Mono-nucleotide shuffle [Default: %default]",
    )
    parser.add_option(
        "--dinuc_shuffle",
        dest="dinuc_shuffle",
        default=False,
        action="store_true",
        help="Di-nucleotide shuffle [Default: %default]",
    )
    parser.add_option(
        "--pseudo",
        dest="pseudo",
        default=None,
        type="float",
        help="Constant pseudo count [Default: %default]",
    )
    parser.add_option(
        "--full_utr",
        dest="full_utr",
        default=False,
        action="store_true",
        help="Run ISM over full UTR [Default: %default]",
    )
    parser.add_option(
        "--apa_file",
        dest="apa_file",
        default=None,
        type="str",
        help="File specifying APA sites",
    )
    parser.add_option(
        "--splice_file",
        dest="splice_file",
        default=None,
        type="str",
        help="File specifying Splice sites",
    )
    parser.add_option(
        "-s",
        dest="start_i",
        default=0,
        type="int",
        help="Gene start index [Default: %default]",
    )
    parser.add_option(
        "-e",
        dest="end_i",
        default=None,
        type="int",
        help="Gene end index [Default: %default]",
    )
    parser.add_option(
        "-p",
        dest="part_i",
        default=0,
        type="int",
        help="Chunk index [Default: %default]",
    )
    parser.add_option(
        "-u",
        dest="untransform_old",
        default=False,
        action="store_true",
    )
    (options, args) = parser.parse_args()

    if len(args) == 3:
        # single worker
        params_file = args[0]
        model_folder = args[1]
        genes_csv_file = args[2]
    else:
        parser.error("Must provide parameter file, model folder and csv file")

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    options.folds = [int(fold) for fold in options.folds.split(",")]
    options.crosses = [int(cross) for cross in options.crosses.split(",")]
    options.shifts = [int(shift) for shift in options.shifts.split(",")]

    #################################################################
    # read parameters and targets

    # read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]
    params_train = params["train"]
    seq_len = params_model["seq_length"]

    if options.targets_file is None:
        parser.error("Must provide targets table to properly handle strands.")
    else:
        targets_df = pd.read_csv(options.targets_file, sep="\t", index_col=0)

    # prep strand
    orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
    targets_strand_pair = np.array([orig_new_index[ti] for ti in targets_df.strand_pair])
    targets_strand_df = targets_prep_strand(targets_df)
    num_targets = len(targets_strand_df)

    # specify relative target indices
    targets_df["row_index"] = np.arange(len(targets_df), dtype="int32")

    #################################################################
    # load first model fold to get parameters

    seqnn_model = seqnn.SeqNN(params_model)
    if options.head_i is not None :
        seqnn_model.restore(model_folder + '/f%dc%d/train/model%d_best.h5' % (options.folds[0], options.crosses[0], options.head_i), options.head_i)
    else :
        seqnn_model.restore(model_folder + '/f%dc%d/train/model_best.h5' % (options.folds[0], options.crosses[0]))
    seqnn_model.build_slice(targets_df.index, False)
    # seqnn_model.build_ensemble(options.rc, options.shifts)

    model_stride = seqnn_model.model_strides[0]
    model_crop = seqnn_model.target_crops[0]
    target_length = seqnn_model.target_lengths[0]

    #################################################################
    # get genome FASTA

    genome_open = pysam.Fastafile(options.genome_fasta)
    
    genome_suffix = ""
    if "hg38" in options.genome_fasta :
        genome_suffix = "_hg38"
    elif "mm10" in options.genome_fasta :
        genome_suffix = "_mm10"
    
    #################################################################
    # gene metadata
    
    # load gene dataframe
    gene_df = pd.read_csv(genes_csv_file, compression="gzip", sep="\t")
    
    # slice current gene chunk
    gene_df = gene_df.iloc[options.start_i:options.end_i].copy().reset_index(drop=True)
    
    # load APA dataframe
    apa_df = pd.read_csv(options.apa_file, compression="gzip", sep="\t")
    if "gtex_blood_pred" in apa_df.columns.values.tolist() :
        apa_df = apa_df.loc[(~apa_df["gtex_blood_pred"].isnull())].copy().reset_index(drop=True)
    elif "rna3_TDP43KD-7d_pred" in apa_df.columns.values.tolist() :
        apa_df = apa_df.loc[(~apa_df["rna3_TDP43KD-7d_pred"].isnull())].copy().reset_index(drop=True)
    apa_df = apa_df.sort_values(by=["chrom", "gene", "site_num_kept"], ascending=True).copy().reset_index(drop=True)

    # index dataframe into dictionary by gene
    gene_apa_dict = {}
    for _, row in apa_df.iterrows():
        if row["gene"] not in gene_apa_dict:
            gene_apa_dict[row["gene"]] = []
        
        gene_apa_dict[row["gene"]].append(row)

    # load splice site annotation
    splice_df = None
    gene_splice_dict = None
    if options.splice_file is not None :
        splice_df = pd.read_csv(options.splice_file, sep="\t", names=["chrom", "havana_str", "feature", "start", "end", "feat1", "strand", "feat2", "id_str"], usecols=["chrom", "havana_str", "feature", "start", "end", "feat1", "strand", "feat2", "id_str"])[["chrom", "havana_str", "feature", "start", "end", "feat1", "strand", "feat2", "id_str"]].query("havana_str == 'HAVANA'").drop_duplicates(subset=["chrom", "start", "strand"], keep="first").copy().reset_index(drop=True)
        
        splice_df = splice_df.loc[splice_df["id_str"].str.contains("gene_name")].copy().reset_index(drop=True)
        splice_df["gene"] = splice_df["id_str"].apply(lambda x: x.split("gene_name \"")[1].split("\";")[0])
        splice_df = splice_df[["chrom", "start", "strand", "gene"]]
        
        # index dataframe into dictionary by gene
        gene_splice_dict = {}
        for _, row in splice_df.iterrows():
            if row["gene"] not in gene_splice_dict:
                gene_splice_dict[row["gene"]] = []
            
            gene_splice_dict[row["gene"]].append(row)

    print("len(gene_df) = " + str(len(gene_df)))
    print("len(gene_apa_dict) = " + str(len(gene_apa_dict)))
    if options.splice_file is not None :
        print("len(gene_splice_dict) = " + str(len(gene_splice_dict)))
    print("len(apa_df) = " + str(len(apa_df)))

    #################################################################
    # setup output

    min_start = -model_stride*model_crop

    # calculate gene start/end sequence coordinates and ism regions
    gene_list = []
    genes_chr = []
    genes_start = []
    genes_end = []
    genes_strand = []
    genes_ism_regions = []
    genes_num_bins = []
    genes_denom_bins = []
    
    # loop over genes
    for gi, [_, row] in enumerate(gene_df.iterrows()):
        gene_list.append(row['gene'])
        genes_chr.append(row['chrom'])
        genes_strand.append(row['strand'])
        gene_midpoint = row['position' + genome_suffix]
        gene_start = max(min_start, gene_midpoint - seq_len // 2)
        gene_end = gene_start + seq_len
        genes_start.append(gene_start)
        genes_end.append(gene_end)
        
        if options.full_utr :
            print('Not implemented for prop3 script.')
        else : # add pA sites and splice junctions to ISM list
            ism_patch_apa = []
            
            # loop over pA sites
            for apa_row in gene_apa_dict[row['gene']] :
                pas_pos = apa_row['position' + genome_suffix] - gene_start
                
                ism_start = pas_pos - options.ism_size // 2
                ism_end = pas_pos + options.ism_size // 2 + options.ism_size % 2
            
                # add to list if within range
                if ism_start - options.window_size // 2 >= 0 and ism_end + options.window_size // 2 + 1 < seq_len :
                    ism_patch_apa.extend(np.arange(ism_start, ism_end + 1, dtype='int32').tolist())
            
            ism_patch_splice = []
            if options.splice_file is not None :
                
                # loop over splice junctions
                for splice_row in gene_splice_dict[row['gene']] :
                    splice_pos = splice_row['start'] - gene_start
                    
                    ism_start = splice_pos - options.splice_ism_size // 2
                    ism_end = splice_pos + options.splice_ism_size // 2 + options.splice_ism_size % 2
                    
                    # add to list if within range
                    if ism_start - options.window_size // 2 >= 0 and ism_end + options.window_size // 2 + 1 < seq_len :
                        ism_patch_splice.extend(np.arange(ism_start, ism_end + 1, dtype='int32').tolist())
            
            ism_patch_splice = [ism_pos for ism_pos in ism_patch_splice if ism_pos >= 0 and ism_pos < seq_len]
            ism_patch = sorted(list(set(ism_patch_apa + ism_patch_splice)))
            
            # extract intervals of contiguous regions
            ism_regions = list(extract_intervals(ism_patch))
            genes_ism_regions.append(ism_regions)
        
        dist_pos = row['position' + genome_suffix] - gene_start
        
        num_bins = []
        denom_bins = []
        
        # loop over pA sites and calculate numerator / denominator bins
        for apa_row_i, apa_row in enumerate(gene_apa_dict[row['gene']]) :
            pas_pos = apa_row['position' + genome_suffix] - gene_start
            
            pas_bin = int(np.round(pas_pos / model_stride)) - model_crop
            
            bin_end = pas_bin + 5
            bin_start = bin_end - 9
            
            # non-distal sites
            if pas_pos != dist_pos :
                denom_bins.extend(np.arange(bin_start, bin_end, dtype='int32').tolist())
            else : # distal sites
                num_bins.extend(np.arange(bin_start, bin_end, dtype='int32').tolist())
        
        num_bins = sorted(list(set(num_bins)))
        denom_bins = sorted(list(set(denom_bins)))
        
        # remove denominator bins that overlap with numerator bins
        denom_bins_dedup = []
        for denom_bin in denom_bins :
            if denom_bin not in num_bins :
                denom_bins_dedup.append(denom_bin)
                
        denom_bins = denom_bins_dedup
        
        num_bins = np.array(num_bins, dtype='int32')
        denom_bins = np.array(denom_bins, dtype='int32')
        
        genes_num_bins.append(num_bins)
        genes_denom_bins.append(denom_bins)
    
    #################################################################
    # predict ISM scores, write output
    
    # scaling parameters
    scales = np.expand_dims(np.ones(len(targets_strand_df), dtype="float32"), axis=0)
    clip_softs = np.expand_dims(np.array(targets_strand_df.clip_soft.values, dtype="float32"), axis=0)
    sqrt_mask = np.expand_dims(np.array([ss.find("sqrt") != -1 for ss in targets_strand_df.sum_stat], dtype=bool), axis=0)
    
    pseudo_cov = None
    if options.pseudo is not None :
        # set pseudo count based on constant value
        pseudo_cov = options.pseudo * np.ones((1, targets_strand_df.shape[0]), dtype="float32") + 1e-6
    elif "pseudo_cov" in targets_strand_df.columns.values.tolist() :
        pseudo_cov = np.expand_dims(np.array(targets_strand_df.pseudo_cov.values, dtype="float32"), axis=0) + 1e-6
    else :
        pseudo_cov = np.zeros((1, targets_strand_df.shape[0]), dtype="float32") + 1e-6
    
    print("Using pseudo_cov = " + str(pseudo_cov[:, :10]), flush=True)
    
    print("n genes = " + str(len(genes_chr)))
    num_genes = len(genes_chr)
    
    # loop over folds
    for fold_ix in options.folds :
        print("-- Fold = " + str(fold_ix) + " --")
        
        # loop over crosses
        for cross_ix in options.crosses :
            print("-- Cross = " + str(cross_ix) + " --")

            # (re-)initialize HDF5
            shuffle_str = ""
            if options.do_shuffle :
                shuffle_str = "_shuffle"

            scores_h5_file = "%s/ism%s_f%dc%d_part%d.h5" % (options.out_dir, shuffle_str, fold_ix, cross_ix, options.part_i)
            if os.path.isfile(scores_h5_file):
                os.remove(scores_h5_file)
            scores_h5 = h5py.File(scores_h5_file, "w")
            scores_h5.create_dataset("seqs", dtype="bool",
                shape=(num_genes, seq_len, 4))
            if not options.separate_rc :
                scores_h5.create_dataset("isms", dtype="float16",
                    shape=(num_genes, seq_len, 4, num_targets))
            else :
                scores_h5.create_dataset("isms", dtype="float16",
                    shape=(num_genes, seq_len, 4, num_targets, 2))
            scores_h5.create_dataset("gene", data=np.array(gene_list, dtype="S"))
            scores_h5.create_dataset("chr", data=np.array(genes_chr, dtype="S"))
            scores_h5.create_dataset("start", data=np.array(genes_start))
            scores_h5.create_dataset("end", data=np.array(genes_end))
            scores_h5.create_dataset("strand", data=np.array(genes_strand, dtype="S"))

            # load model fold
            seqnn_model = seqnn.SeqNN(params_model)
            if options.head_i is not None :
                seqnn_model.restore(model_folder + '/f%dc%d/train/model%d_best.h5' % (fold_ix, cross_ix, options.head_i), options.head_i)
            else :
                seqnn_model.restore(model_folder + '/f%dc%d/train/model_best.h5' % (fold_ix, cross_ix))
            seqnn_model.build_slice(targets_df.index, False)

            for shift in options.shifts :
                print("Processing shift %d" % shift, flush=True)

                for rev_comp in ([False, True] if options.rc else [False]) :

                    if options.rc :
                        print("Fwd/rev = %s" % ("fwd" if not rev_comp else "rev"), flush=True)

                    for gi, gene_id in enumerate(gene_list):

                        if gi % 1 == 0 :
                            print("Processing %d, %s" % (gi, gene_id), flush=True)

                        # make sequence
                        seq_1hot = make_seq_1hot(genome_open, genes_chr[gi], genes_start[gi], genes_end[gi], seq_len)
                        seq_1hot = dna_io.hot1_augment(seq_1hot, shift=shift)

                        # determine output positions
                        gene_slice = genes_num_bins[gi]
                        gene_slice_denom = genes_denom_bins[gi]

                        # get ism window regions
                        gene_ism_regions = genes_ism_regions[gi]

                        if rev_comp:
                            seq_1hot = dna_io.hot1_rc(seq_1hot)
                            gene_slice = target_length - gene_slice - 1
                            gene_slice_denom = target_length - gene_slice_denom - 1

                            gene_ism_regions = []
                            for [genes_ism_start_orig, gene_ism_end_orig] in genes_ism_regions[gi] :
                                gene_ism_start = seq_len - gene_ism_end_orig
                                gene_ism_end = seq_len - genes_ism_start_orig

                                gene_ism_regions.append([gene_ism_start, gene_ism_end])

                        # slice relevant strand targets
                        if genes_strand[gi] == "+":
                            gene_strand_mask = (targets_df.strand != "-") if not rev_comp else (targets_df.strand != "+")
                        else:
                            gene_strand_mask = (targets_df.strand != "+") if not rev_comp else (targets_df.strand != "-")

                        gene_target = np.array(targets_df.index[gene_strand_mask].values)

                        # broadcast to singleton batch
                        seq_1hot = seq_1hot[None, ...]
                        gene_slice = gene_slice[None, ...]
                        gene_target = gene_target[None, ...]

                        # ism computation
                        ism = None
                        if options.do_shuffle :
                            ism = get_ism_shuffle(
                                seqnn_model,
                                seq_1hot,
                                gene_ism_regions,
                                head_i=0,
                                target_slice=gene_target,
                                pos_slice=gene_slice,
                                pos_slice_denom=gene_slice_denom,
                                sqrt_mask=sqrt_mask,
                                scales=scales,
                                clip_softs=clip_softs,
                                pseudo_counts=pseudo_cov,
                                untransform_old=options.untransform_old,
                                use_mean=False,
                                use_ratio=True,
                                use_logodds=False,
                                window_size=options.window_size,
                                n_samples=options.n_samples,
                                mononuc_shuffle=options.mononuc_shuffle,
                                dinuc_shuffle=options.dinuc_shuffle,
                            )
                        else :
                            ism = get_ism(
                                seqnn_model,
                                seq_1hot,
                                gene_ism_regions,
                                head_i=0,
                                target_slice=gene_target,
                                pos_slice=gene_slice,
                                pos_slice_denom=gene_slice_denom,
                                sqrt_mask=sqrt_mask,
                                scales=scales,
                                clip_softs=clip_softs,
                                pseudo_counts=pseudo_cov,
                                untransform_old=options.untransform_old,
                                use_mean=False,
                                use_ratio=True,
                                use_logodds=False,
                            )

                        # undo augmentations and save ism
                        ism = unaugment_grads(ism, fwdrc=(not rev_comp), shift=shift)

                        # write to HDF5
                        if not options.separate_rc :
                            scores_h5["isms"][gi] += ism[:, ...]
                        elif not rev_comp :
                            scores_h5["isms"][gi, ..., 0] += ism[:, ...]
                        else :
                            scores_h5["isms"][gi, ..., 1] += ism[:, ...]

                        # collect garbage
                        gc.collect()

            # save sequences and normalize isms by total size of ensemble
            for gi, gene_id in enumerate(gene_list):

                # re-make original sequence
                seq_1hot = make_seq_1hot(genome_open, genes_chr[gi], genes_start[gi], genes_end[gi], seq_len)

                # write to HDF5
                scores_h5["seqs"][gi] = seq_1hot[:, ...]

                if not options.separate_rc :
                    scores_h5["isms"][gi] /= float((len(options.shifts) * (2 if options.rc else 1)))
                else :
                    scores_h5["isms"][gi, ..., 0] /= float((len(options.shifts)))
                    scores_h5["isms"][gi, ..., 1] /= float((len(options.shifts)))

            # collect garbage
            gc.collect()

            scores_h5.close()

            # write to file indicating successful completion
            success_file = "%s/ism%s_f%dc%d_part%d-success.txt" % (options.out_dir, shuffle_str, fold_ix, cross_ix, options.part_i)

            with open(success_file, "wt") as f :
                f.write("success\n")

    # close files
    genome_open.close()


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
        seq_dna = "N"*(-start) + genome_open.fetch(chrm, 0, end)
    else:
        seq_dna = genome_open.fetch(chrm, start, end)
        
    # extend to full length
    if len(seq_dna) < seq_len:
        seq_dna += "N"*(seq_len-len(seq_dna))

    seq_1hot = dna_io.dna_1hot(seq_dna)
    return seq_1hot


# tf code for computing ISM scores on GPU
@tf.function
def _score_func(model, seq_1hot, target_slice, pos_slice, pos_mask=None, pos_slice_denom=None, pos_mask_denom=True, sqrt_mask=None, scales=None, clip_softs=None, pseudo_counts=None, untransform_old=False, use_mean=False, use_ratio=False, use_logodds=False) :
            
    # predict
    preds = tf.gather(model(seq_1hot, training=False), target_slice, axis=-1, batch_dims=1)

    if sqrt_mask is not None and scales is not None and clip_softs is not None :
        if untransform_old:
            # undo scale
            preds = preds / scales
            
            # undo clipsoft
            preds = tf.where(preds > clip_softs, (preds - clip_softs) ** 2 + clip_softs, preds)

            # undo sqrt
            preds = tf.where(sqrt_mask, preds ** (4/3), preds)
        else:
            # undo clipsoft
            preds = tf.where(preds > clip_softs, (preds - clip_softs + 1)**2 + clip_softs - 1, preds)

            # undo sqrt
            preds = tf.where(sqrt_mask, (preds + 1)**(4/3) - 1, preds)

            # undo scale
            preds = preds / scales

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
    if not use_ratio :
        if pseudo_counts is not None :
            score_ratios = tf.math.log(preds_agg + pseudo_counts)
        else :
            score_ratios = tf.math.log(preds_agg + 1e-6)
    else :
        if not use_logodds :
            if pseudo_counts is not None :
                score_ratios = tf.math.log((preds_agg + pseudo_counts) / (preds_agg_denom + pseudo_counts))
            else :
                score_ratios = tf.math.log((preds_agg + 1e-6) / (preds_agg_denom + 1e-6))
        else :
            if pseudo_counts is not None :
                score_ratios = tf.math.log(((preds_agg + pseudo_counts) / (preds_agg_denom + pseudo_counts)) / (1. - ((preds_agg + pseudo_counts) / (preds_agg_denom + pseudo_counts))))
            else :
                score_ratios = tf.math.log(((preds_agg + 1e-6) / (preds_agg_denom + 1e-6)) / (1. - ((preds_agg + 1e-6) / (preds_agg_denom + 1e-6))))

    return score_ratios

# calculate ism window shuffle
def get_ism_shuffle(seqnn_model, seq_1hot_wt, ism_regions, head_i=None, target_slice=None, pos_slice=None, pos_mask=None, pos_slice_denom=None, pos_mask_denom=None, sqrt_mask=None, scales=None, clip_softs=None, pseudo_counts=None, untransform_old=False, use_mean=False, use_ratio=False, use_logodds=False, bases=[0, 1, 2, 3], window_size=5, n_samples=8, mononuc_shuffle=False, dinuc_shuffle=False) :
        
    # choose model
    if seqnn_model.ensemble is not None:
        model = seqnn_model.ensemble
    elif head_i is not None:
        model = seqnn_model.models[head_i]
    else:
        model = seqnn_model.model

    # verify tensor shape(s)
    seq_1hot_wt = seq_1hot_wt.astype("float32")
    target_slice = np.array(target_slice).astype("int32")
    pos_slice = np.array(pos_slice).astype("int32")

    # convert constants to tf tensors
    if sqrt_mask is not None :
        sqrt_mask = tf.convert_to_tensor(sqrt_mask, dtype=tf.bool)
    if scales is not None :
        scales = tf.convert_to_tensor(scales, dtype=tf.float32)
    if clip_softs is not None :
        clip_softs = tf.convert_to_tensor(clip_softs, dtype=tf.float32)
    if pseudo_counts is not None :
        pseudo_counts = tf.convert_to_tensor(pseudo_counts, dtype=tf.float32)

    if pos_mask is not None :
        pos_mask = np.array(pos_mask).astype("float32")

    if use_ratio and pos_slice_denom is not None :
        pos_slice_denom = np.array(pos_slice_denom).astype("int32")

        if pos_mask_denom is not None :
            pos_mask_denom = np.array(pos_mask_denom).astype("float32")

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
    pred_shuffle = np.zeros((seq_1hot_wt.shape[1], n_samples, target_slice.shape[1]))

    # get wt pred
    score_wt = _score_func(model, seq_1hot_wt_tf, target_slice, pos_slice, pos_mask, pos_slice_denom, pos_mask_denom, sqrt_mask, scales, clip_softs, pseudo_counts, untransform_old, use_mean, use_ratio, use_logodds).numpy()

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
                else: # dinuc-shuffle
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
                score_mut = _score_func(model, seq_1hot_mut_tf, target_slice, pos_slice, pos_mask, pos_slice_denom, pos_mask_denom, sqrt_mask, scales, clip_softs, pseudo_counts, untransform_old, use_mean, use_ratio, use_logodds).numpy()

                pred_shuffle[j, sample_ix, :] = score_wt - score_mut

    pred_ism = np.tile(np.mean(pred_shuffle, axis=1, keepdims=True), (1, 4, 1)) * seq_1hot_wt[0, ..., None]

    return pred_ism

# calculate ism
def get_ism(seqnn_model, seq_1hot_wt, ism_regions, head_i=None, target_slice=None, pos_slice=None, pos_mask=None, pos_slice_denom=None, pos_mask_denom=None, sqrt_mask=None, scales=None, clip_softs=None, pseudo_counts=None, untransform_old=False, use_mean=False, use_ratio=False, use_logodds=False, bases=[0, 1, 2, 3]) :
        
    # choose model
    if seqnn_model.ensemble is not None:
        model = seqnn_model.ensemble
    elif head_i is not None:
        model = seqnn_model.models[head_i]
    else:
        model = seqnn_model.model

    # verify tensor shape(s)
    seq_1hot_wt = seq_1hot_wt.astype("float32")
    target_slice = np.array(target_slice).astype("int32")
    pos_slice = np.array(pos_slice).astype("int32")

    # convert constants to tf tensors
    if sqrt_mask is not None :
        sqrt_mask = tf.convert_to_tensor(sqrt_mask, dtype=tf.bool)
    if scales is not None :
        scales = tf.convert_to_tensor(scales, dtype=tf.float32)
    if clip_softs is not None :
        clip_softs = tf.convert_to_tensor(clip_softs, dtype=tf.float32)
    if pseudo_counts is not None :
        pseudo_counts = tf.convert_to_tensor(pseudo_counts, dtype=tf.float32)

    if pos_mask is not None :
        pos_mask = np.array(pos_mask).astype("float32")

    if use_ratio and pos_slice_denom is not None :
        pos_slice_denom = np.array(pos_slice_denom).astype("int32")

        if pos_mask_denom is not None :
            pos_mask_denom = np.array(pos_mask_denom).astype("float32")

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

    # allocate ism result tensor
    pred_ism = np.zeros((524288, 4, target_slice.shape[1]))

    # get wt pred
    score_wt = _score_func(model, seq_1hot_wt_tf, target_slice, pos_slice, pos_mask, pos_slice_denom, pos_mask_denom, sqrt_mask, scales, clip_softs, pseudo_counts, untransform_old, use_mean, use_ratio, use_logodds).numpy()

    for ism_region_i, [ism_start, ism_end] in enumerate(ism_regions) :
        for j in range(ism_start, ism_end) :
            for b in bases :
                if seq_1hot_wt[0, j, b] != 1. : 
                    seq_1hot_mut = np.copy(seq_1hot_wt)
                    seq_1hot_mut[0, j, :] = 0.
                    seq_1hot_mut[0, j, b] = 1.

                    # convert to tf tensor
                    seq_1hot_mut_tf = tf.convert_to_tensor(seq_1hot_mut, dtype=tf.float32)

                    # get mut pred
                    score_mut = _score_func(model, seq_1hot_mut_tf, target_slice, pos_slice, pos_mask, pos_slice_denom, pos_mask_denom, sqrt_mask, scales, clip_softs, pseudo_counts, untransform_old, use_mean, use_ratio, use_logodds).numpy()

                    pred_ism[j, b, :] = score_wt - score_mut

    pred_ism = np.tile(np.mean(pred_ism, axis=1, keepdims=True), (1, 4, 1)) * seq_1hot_wt[0, ..., None]

    return pred_ism

# extract intervals of contiguous ranges
def extract_intervals(x):
    x_it = sorted(set(x))

    # loop over intervals
    for key, group in itertools.groupby(enumerate(x_it), lambda t: t[1] - t[0]):
        group = list(group)
        yield [group[0][1], group[-1][1]]

################################################################################
# __main__
# ###############################################################################
if __name__ == "__main__":
    main()
