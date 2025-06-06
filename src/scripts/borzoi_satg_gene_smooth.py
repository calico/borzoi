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
import time

import h5py
import numpy as np
import pandas as pd
import pysam
import pybedtools
import tensorflow as tf

from baskerville.dataset import targets_prep_strand
from baskerville import dna as dna_io
from baskerville import gene as bgene
from baskerville import seqnn

"""
borzoi_satg_gene_smooth.py

Perform a smoothed gradient saliency analysis for genes specified in a GTF file (integrating over mutated sequence samples).
"""

################################################################################
# main
# ###############################################################################
def main():
    usage = "usage: %prog [options] <params> <model> <gene_gtf>"
    parser = OptionParser(usage)
    parser.add_option(
        "--fa",
        dest="genome_fasta",
        default="%s/assembly/ucsc/hg38.fa" % os.environ.get('BORZOI_HG38', 'hg38'),
        help="Genome FASTA for sequences [Default: %default]",
    )
    parser.add_option(
        "--full_gtf",
        dest="full_gtf",
        default="%s/genes/gencode41/gencode41_basic_nort_protein.gtf" % os.environ.get('BORZOI_HG38', 'hg38'),
        help="Full genes gtf file [Default: %default]",
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
        "-c",
        dest="crosses",
        default="0",
        type="str",
        help="Model crosses (replicates) to use in ensemble (comma-separated list) [Default:%default]",
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
        "--samples",
        dest="n_samples",
        default=5,
        type="int",
        help="Number of smoothgrad samples [Default: %default]",
    )
    parser.add_option(
        "--sampleprob",
        dest="sample_prob",
        default=0.90,
        type="float",
        help="Probability of not mutating a position in smoothgrad [Default: %default]",
    )
    parser.add_option(
        "--sampleval",
        dest="sample_value",
        default=1.0,
        type="float",
        help="New one-hot value for a corrupted position in smoothgrad [Default: %default]",
    )
    parser.add_option(
        "--sampleseed",
        dest="sample_seed",
        default=42,
        type="int",
        help="Smoothgrad random seed [Default: %default]",
    )
    parser.add_option(
        "--restrict_exons",
        dest="restrict_exons",
        default=False,
        action="store_true",
        help="Do not mutate exons of target gene [Default: %default]",
    )
    parser.add_option(
        "--restrict_other_exons",
        dest="restrict_other_exons",
        default=False,
        action="store_true",
        help="Do not mutate exons of other genes [Default: %default]",
    )
    parser.add_option(
        "--exon_padding_bp",
        dest="exon_padding_bp",
        default=48,
        type="int",
        help="Do not mutate within this radius of an exon junction [Default: %default]",
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
        "--get_preds",
        dest="get_preds",
        default=False,
        action="store_true",
        help="Store scalar predictions in addition to their gradients [Default: %default]",
    )
    parser.add_option(
        "--pseudo",
        dest="pseudo",
        default=None,
        type="float",
        help="Constant pseudo count [Default: %default]",
    )
    parser.add_option(
        "--pseudo_qtl",
        dest="pseudo_qtl",
        default=None,
        type="float",
        help="Quantile of predicted scalars to choose as pseudo count [Default: %default]",
    )
    parser.add_option(
        "--pseudo_tissue",
        dest="pseudo_tissue",
        default=None,
        type="str",
        help="Tissue to filter genes on when calculating pseudo count [Default: %default]",
    )
    parser.add_option(
        "--gene_file",
        dest="gene_file",
        default=None,
        type="str",
        help="Csv-file of gene metadata [Default: %default]",
    )
    parser.add_option(
        "-t",
        dest="targets_file",
        default=None,
        type="str",
        help="File specifying target indexes and labels in table format",
    )
    (options, args) = parser.parse_args()

    if len(args) == 3:
        # single worker
        params_file = args[0]
        model_folder = args[1]
        genes_gtf_file = args[2]
    else:
        parser.error("Must provide parameter file, model folder and GTF file")

    if not os.path.isdir(options.out_dir):
        os.makedirs(options.out_dir, exist_ok=True)

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
    targets_strand_pair = np.array(
        [orig_new_index[ti] for ti in targets_df.strand_pair]
    )
    targets_strand_df = targets_prep_strand(targets_df)
    num_targets = 1

    # load gene dataframe and select tissue
    tissue_genes = None
    if options.gene_file is not None and options.pseudo_tissue is not None:
        gene_df = pd.read_csv(options.gene_file, sep="\t")
        gene_df = (
            gene_df.query("tissue == '" + str(options.pseudo_tissue) + "'")
            .copy()
            .reset_index(drop=True)
        )
        gene_df = gene_df.drop(columns=["Unnamed: 0"])

        # get list of gene for tissue
        tissue_genes = gene_df["gene_base"].values.tolist()

        print("len(tissue_genes) = " + str(len(tissue_genes)))
    
    # optionally set pseudo count
    pseudo_count = 0.0
    if options.pseudo is not None :
        # set pseudo count based on constant
        pseudo_count = options.pseudo
    elif 'pseudo' in targets_strand_df.columns.values.tolist() :
        pseudo_count = round(np.mean(targets_strand_df['pseudo'].values), 6)

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
    full_transcriptome = bgene.Transcriptome(options.full_gtf)
  
    #Get gene span bedtool (of full transcriptome)
    bedt_span = full_transcriptome.bedtool_span()

    # order valid genes
    genome_open = pysam.Fastafile(options.genome_fasta)
    gene_list = sorted(transcriptome.genes.keys())
    num_genes = len(gene_list)

    #################################################################
    # setup output

    min_start = -model_stride * model_crop

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
        gene_start = max(min_start, gene_midpoint - seq_len // 2)
        gene_end = gene_start + seq_len
        genes_start.append(gene_start)
        genes_end.append(gene_end)

    #################################################################
    # predict scores, write output

    buffer_size = 1024

    print("clip_soft = " + str(options.clip_soft))

    print("n genes = " + str(len(genes_chr)))

    # loop over folds
    for fold_ix in options.folds:
        for cross_ix in options.crosses:
            
            print("-- fold = f" + str(fold_ix) + "c" + str(cross_ix) + " --")

            # (re-)initialize HDF5
            scores_h5_file = "%s/scores_f%dc%d.h5" % (options.out_dir, fold_ix, cross_ix)
            if os.path.isfile(scores_h5_file):
                os.remove(scores_h5_file)
            scores_h5 = h5py.File(scores_h5_file, "w")
            scores_h5.create_dataset("seqs", dtype="bool", shape=(num_genes, seq_len, 4))
            scores_h5.create_dataset(
                "grads", dtype="float16", shape=(num_genes, seq_len, 4, num_targets)
            )
            if options.get_preds:
                scores_h5.create_dataset(
                    "preds", dtype="float32", shape=(num_genes, num_targets)
                )
            if options.restrict_exons or options.restrict_other_exons :
                scores_h5.create_dataset(
                    "masks", dtype="bool", shape=(num_genes, seq_len)
                )
            scores_h5.create_dataset("gene", data=np.array(gene_list, dtype="S"))
            scores_h5.create_dataset("chr", data=np.array(genes_chr, dtype="S"))
            scores_h5.create_dataset("start", data=np.array(genes_start))
            scores_h5.create_dataset("end", data=np.array(genes_end))
            scores_h5.create_dataset("strand", data=np.array(genes_strand, dtype="S"))

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

            # optionally get (and store) scalar predictions before computing their gradients
            if options.get_preds:
                print(" - (prediction) - ", flush=True)

                for shift in options.shifts:
                    print("Processing shift %d" % shift, flush=True)

                    for rev_comp in [False, True] if options.rc else [False]:

                        if options.rc:
                            print(
                                "Fwd/rev = %s" % ("fwd" if not rev_comp else "rev"),
                                flush=True,
                            )

                        seq_1hots = []
                        gene_slices = []
                        gene_targets = []

                        for gi, gene_id in enumerate(gene_list):

                            if gi % 500 == 0:
                                print("Processing %d, %s" % (gi, gene_id), flush=True)

                            gene = transcriptome.genes[gene_id]

                            # make sequence
                            seq_1hot = make_seq_1hot(
                                genome_open,
                                genes_chr[gi],
                                genes_start[gi],
                                genes_end[gi],
                                seq_len,
                            )
                            seq_1hot = dna_io.hot1_augment(seq_1hot, shift=shift)

                            # determine output sequence start
                            seq_out_start = genes_start[gi] + model_stride * model_crop
                            seq_out_len = model_stride * target_length

                            # determine output positions
                            gene_slice = gene.output_slice(
                                seq_out_start, seq_out_len, model_stride, options.span
                            )

                            if rev_comp:
                                seq_1hot = dna_io.hot1_rc(seq_1hot)
                                gene_slice = target_length - gene_slice - 1

                            # slice relevant strand targets
                            if genes_strand[gi] == "+":
                                gene_strand_mask = (
                                    (targets_df.strand != "-")
                                    if not rev_comp
                                    else (targets_df.strand != "+")
                                )
                            else:
                                gene_strand_mask = (
                                    (targets_df.strand != "+")
                                    if not rev_comp
                                    else (targets_df.strand != "-")
                                )

                            gene_target = np.array(
                                targets_df.index[gene_strand_mask].values
                            )

                            # accumulate data tensors
                            seq_1hots.append(seq_1hot[None, ...])
                            gene_slices.append(gene_slice[None, ...])
                            gene_targets.append(gene_target[None, ...])

                            if gi == len(gene_list) - 1 or len(seq_1hots) >= buffer_size:

                                # concat sequences
                                seq_1hots = np.concatenate(seq_1hots, axis=0)

                                # pad gene slices to same length (mark valid positions in mask tensor)
                                max_slice_len = int(
                                    np.max(
                                        [gene_slice.shape[1] for gene_slice in gene_slices]
                                    )
                                )

                                gene_masks = np.zeros(
                                    (len(gene_slices), max_slice_len), dtype="float32"
                                )
                                gene_slices_padded = np.zeros(
                                    (len(gene_slices), max_slice_len), dtype="int32"
                                )
                                for gii, gene_slice in enumerate(gene_slices):
                                    for j in range(gene_slice.shape[1]):
                                        gene_masks[gii, j] = 1.0
                                        gene_slices_padded[gii, j] = gene_slice[0, j]

                                gene_slices = gene_slices_padded

                                # concat gene-specific targets
                                gene_targets = np.concatenate(gene_targets, axis=0)

                                # batch call count predictions
                                preds = predict_counts(
                                    seqnn_model,
                                    seq_1hots,
                                    head_i=0,
                                    target_slice=gene_targets,
                                    pos_slice=gene_slices,
                                    pos_mask=gene_masks,
                                    chunk_size=buffer_size,
                                    batch_size=1,
                                    track_scale=options.track_scale,
                                    track_transform=options.track_transform,
                                    clip_soft=options.clip_soft,
                                    pseudo_count=pseudo_count,
                                    untransform_old=options.untransform_old,
                                    use_mean=False,
                                    dtype="float32",
                                )

                                # save predictions
                                for gii, gene_slice in enumerate(gene_slices):
                                    h5_gi = (gi // buffer_size) * buffer_size + gii

                                    # write to HDF5
                                    scores_h5["preds"][h5_gi, :] += preds[gii] / float(
                                        (len(options.shifts) * (2 if options.rc else 1))
                                    )

                                # clear sequence buffer
                                seq_1hots = []
                                gene_slices = []
                                gene_targets = []

                                # collect garbage
                                gc.collect()

            # optionally set pseudo count from predictions
            if options.pseudo_qtl is not None:
                gene_preds = scores_h5["preds"][:]

                # filter on tissue
                tissue_preds = None

                if tissue_genes is not None:
                    tissue_set = set(tissue_genes)

                    # get subset of genes and predictions belonging to the pseudo count tissue
                    tissue_preds = []
                    for gi, gene_id in enumerate(gene_list):
                        if gene_id.split(".")[0] in tissue_set:
                            tissue_preds.append(gene_preds[gi, 0])

                    tissue_preds = np.array(tissue_preds, dtype="float32")
                else:
                    tissue_preds = np.array(gene_preds[:, 0], dtype="float32")

                # set pseudo count based on quantile of predictions
                pseudo_count = np.quantile(tissue_preds, q=options.pseudo_qtl)

                print("")
                print("pseudo_count = " + str(round(pseudo_count, 6)))

            # compute gradients
            print(" - (gradients) - ", flush=True)

            for shift in options.shifts:
                print("Processing shift %d" % shift, flush=True)

                for rev_comp in [False, True] if options.rc else [False]:

                    if options.rc:
                        print(
                            "Fwd/rev = %s" % ("fwd" if not rev_comp else "rev"), flush=True
                        )

                    seq_1hots = []
                    gene_slices = []
                    gene_targets = []
        
                    sample_masks = None
                    if options.restrict_exons or options.restrict_other_exons :
                        sample_masks = []

                    for gi, gene_id in enumerate(gene_list):

                        if gi % 500 == 0:
                            print("Processing %d, %s" % (gi, gene_id), flush=True)

                        gene = transcriptome.genes[gene_id]

                        # make sequence
                        seq_1hot = make_seq_1hot(
                            genome_open,
                            genes_chr[gi],
                            genes_start[gi],
                            genes_end[gi],
                            seq_len,
                        )
                        seq_1hot = dna_io.hot1_augment(seq_1hot, shift=shift)

                        # determine output sequence start
                        seq_out_start = genes_start[gi] + model_stride * model_crop
                        seq_out_len = model_stride * target_length

                        # determine output positions
                        gene_slice = gene.output_slice(
                            seq_out_start, seq_out_len, model_stride, options.span
                        )

                        if rev_comp:
                            seq_1hot = dna_io.hot1_rc(seq_1hot)
                            gene_slice = target_length - gene_slice - 1

                        # slice relevant strand targets
                        if genes_strand[gi] == "+":
                            gene_strand_mask = (
                                (targets_df.strand != "-")
                                if not rev_comp
                                else (targets_df.strand != "+")
                            )
                        else:
                            gene_strand_mask = (
                                (targets_df.strand != "+")
                                if not rev_comp
                                else (targets_df.strand != "-")
                            )

                        gene_target = np.array(targets_df.index[gene_strand_mask].values)

                        if options.restrict_exons or options.restrict_other_exons :

                            sample_mask = np.ones(seq_len, dtype=bool)

                            # restrict exon-covered bases of target gene
                            if options.restrict_exons :

                                # determine exon-overlapping positions
                                this_slice = gene.output_slice(genes_start[gi], seq_len, model_stride, options.span)

                                if rev_comp:
                                    this_slice = seq_len // model_stride - this_slice - 1

                                for bin_ix in this_slice.tolist() :
                                    sample_mask[bin_ix * model_stride - options.exon_padding_bp:(bin_ix+1) * model_stride + options.exon_padding_bp + 1] = False

                            # restrict exon-covered bases of other genes
                            if options.restrict_other_exons :
                                # get sequence bedtool
                                seq_bedt = pybedtools.BedTool('%s %d %d' % (genes_chr[gi], max(genes_start[gi], 0), genes_end[gi]), from_string=True)

                                gene_ids = sorted(list(set([overlap[3] for overlap in bedt_span.intersect(seq_bedt, wo=True) if gene_id not in overlap[3]])))
                                for other_gene_id in gene_ids :

                                    other_slice = full_transcriptome.genes[other_gene_id].output_slice(genes_start[gi], seq_len, model_stride, options.span)

                                    if rev_comp:
                                        other_slice = seq_len // model_stride - other_slice - 1

                                    for bin_ix in other_slice.tolist() :
                                        sample_mask[bin_ix * model_stride - options.exon_padding_bp:(bin_ix+1) * model_stride + options.exon_padding_bp + 1] = False

                        # accumulate data tensors
                        seq_1hots.append(seq_1hot[None, ...])
                        gene_slices.append(gene_slice[None, ...])
                        gene_targets.append(gene_target[None, ...])
          
                        if options.restrict_exons or options.restrict_other_exons :
                            sample_masks.append(sample_mask[None, ...])

                        if gi == len(gene_list) - 1 or len(seq_1hots) >= buffer_size:

                            # concat sequences
                            seq_1hots = np.concatenate(seq_1hots, axis=0)

                            # pad gene slices to same length (mark valid positions in mask tensor)
                            max_slice_len = int(
                                np.max([gene_slice.shape[1] for gene_slice in gene_slices])
                            )

                            gene_masks = np.zeros(
                                (len(gene_slices), max_slice_len), dtype="float32"
                            )
                            gene_slices_padded = np.zeros(
                                (len(gene_slices), max_slice_len), dtype="int32"
                            )
                            for gii, gene_slice in enumerate(gene_slices):
                                for j in range(gene_slice.shape[1]):
                                    gene_masks[gii, j] = 1.0
                                    gene_slices_padded[gii, j] = gene_slice[0, j]

                            gene_slices = gene_slices_padded

                            # concat gene-specific targets
                            gene_targets = np.concatenate(gene_targets, axis=0)
            
                            # concat gene-specific sample mask (for smooth grad)
                            if options.restrict_exons or options.restrict_other_exons :
                                sample_masks = np.concatenate(sample_masks, axis=0)

                            # batch call gradient computation
                            grads = seqnn_model.smooth_gradients(
                                seq_1hots,
                                head_i=0,
                                target_slice=gene_targets,
                                pos_slice=gene_slices,
                                pos_mask=gene_masks,
                                chunk_size=buffer_size // options.n_samples,
                                batch_size=1,
                                track_scale=options.track_scale,
                                track_transform=options.track_transform,
                                clip_soft=options.clip_soft,
                                pseudo_count=pseudo_count,
                                untransform_old=options.untransform_old,
                                no_untransform=options.no_untransform,
                                use_mean=False,
                                use_ratio=False,
                                use_logodds=False,
                                subtract_avg=True,
                                input_gate=False,
                                n_samples=options.n_samples,
                                sample_prob=options.sample_prob,
                                sample_mask=sample_masks,
                                sample_value=options.sample_value,
                                sample_seed=options.sample_seed,
                                dtype="float16",
                            )

                            # undo augmentations and save gradients
                            for gii, gene_slice in enumerate(gene_slices):
                                grad = unaugment_grads(
                                    grads[gii, :, :, None],
                                    fwdrc=(not rev_comp),
                                    shift=shift,
                                )

                                h5_gi = (gi // buffer_size) * buffer_size + gii

                                # write to HDF5
                                scores_h5["grads"][h5_gi] += grad
              
                                if not rev_comp and shift == 0 and (options.restrict_exons or options.restrict_other_exons) :
                                    scores_h5['masks'][h5_gi, :] = sample_masks[gii]

                            # clear sequence buffer
                            seq_1hots = []
                            gene_slices = []
                            gene_targets = []
                            if options.restrict_exons or options.restrict_other_exons :
                                sample_masks = []

                            # collect garbage
                            gc.collect()

            # save sequences and normalize gradients by total size of ensemble
            for gi, gene_id in enumerate(gene_list):

                # re-make original sequence
                seq_1hot = make_seq_1hot(
                    genome_open, genes_chr[gi], genes_start[gi], genes_end[gi], seq_len
                )

                # write to HDF5
                scores_h5["seqs"][gi] = seq_1hot
                scores_h5["grads"][gi] /= float(
                    (len(options.shifts) * (2 if options.rc else 1))
                )

            # collect garbage
            gc.collect()

    # close files
    genome_open.close()
    scores_h5.close()


def unaugment_grads(grads, fwdrc=False, shift=0):
    """Undo sequence augmentation."""
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
        seq_dna = "N" * (-start) + genome_open.fetch(chrm, 0, end)
    else:
        seq_dna = genome_open.fetch(chrm, start, end)

    # extend to full length
    if len(seq_dna) < seq_len:
        seq_dna += "N" * (seq_len - len(seq_dna))

    seq_1hot = dna_io.dna_1hot(seq_dna)
    return seq_1hot


# tf code for predicting raw sum-of-expression counts on GPU
@tf.function
def _count_func(
    model,
    seq_1hot,
    target_slice,
    pos_slice,
    pos_mask=None,
    track_scale=1.0,
    track_transform=1.0,
    clip_soft=None,
    pseudo_count=0.0,
    untransform_old=False,
    use_mean=False,
):

    # predict
    preds = tf.gather(
        model(seq_1hot, training=False), target_slice, axis=-1, batch_dims=1
    )

    if untransform_old:
        # undo scale
        preds = preds / track_scale

        # undo clip_soft
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
    
    # aggregate over tracks (average)
    preds = tf.reduce_mean(preds, axis=-1)

    # slice specified positions
    preds_slice = tf.gather(preds, pos_slice, axis=-1, batch_dims=1)
    if pos_mask is not None:
        preds_slice = preds_slice * pos_mask

    # aggregate over positions
    if not use_mean:
        preds_agg = tf.reduce_sum(preds_slice, axis=-1)
    else:
        if pos_mask is not None:
            preds_agg = tf.reduce_sum(preds_slice, axis=-1) / tf.reduce_sum(
                pos_mask, axis=-1
            )
        else:
            preds_agg = tf.reduce_mean(preds_slice, axis=-1)

    return preds_agg + pseudo_count


# code for getting model predictions from a tensor of input sequence patterns
def predict_counts(
    seqnn_model,
    seq_1hot,
    head_i=None,
    target_slice=None,
    pos_slice=None,
    pos_mask=None,
    chunk_size=None,
    batch_size=1,
    track_scale=1.0,
    track_transform=1.0,
    clip_soft=None,
    pseudo_count=0.0,
    untransform_old=False,
    use_mean=False,
    dtype="float32",
):

    # start time
    t0 = time.time()

    # choose model
    if seqnn_model.ensemble is not None:
        model = seqnn_model.ensemble
    elif head_i is not None:
        model = seqnn_model.models[head_i]
    else:
        model = seqnn_model.model

    # verify tensor shape(s)
    seq_1hot = seq_1hot.astype("float32")
    target_slice = np.array(target_slice).astype("int32")
    pos_slice = np.array(pos_slice).astype("int32")

    # convert constants to tf tensors
    track_scale = tf.constant(track_scale, dtype=tf.float32)
    track_transform = tf.constant(track_transform, dtype=tf.float32)
    if clip_soft is not None:
        clip_soft = tf.constant(clip_soft, dtype=tf.float32)
    
    pseudo_count = tf.constant(pseudo_count, dtype=tf.float32)

    if pos_mask is not None:
        pos_mask = np.array(pos_mask).astype("float32")

    if len(seq_1hot.shape) < 3:
        seq_1hot = seq_1hot[None, ...]

    if len(target_slice.shape) < 2:
        target_slice = target_slice[None, ...]

    if len(pos_slice.shape) < 2:
        pos_slice = pos_slice[None, ...]

    if pos_mask is not None and len(pos_mask.shape) < 2:
        pos_mask = pos_mask[None, ...]

    # chunk parameters
    num_chunks = 1
    if chunk_size is None:
        chunk_size = seq_1hot.shape[0]
    else:
        num_chunks = int(np.ceil(seq_1hot.shape[0] / chunk_size))

    # loop over chunks
    pred_chunks = []
    for ci in range(num_chunks):

        # collect chunk
        seq_1hot_chunk = seq_1hot[ci * chunk_size : (ci + 1) * chunk_size, ...]
        target_slice_chunk = target_slice[ci * chunk_size : (ci + 1) * chunk_size, ...]
        pos_slice_chunk = pos_slice[ci * chunk_size : (ci + 1) * chunk_size, ...]

        pos_mask_chunk = None
        if pos_mask is not None:
            pos_mask_chunk = pos_mask[ci * chunk_size : (ci + 1) * chunk_size, ...]

        actual_chunk_size = seq_1hot_chunk.shape[0]

        # convert to tf tensors
        seq_1hot_chunk = tf.convert_to_tensor(seq_1hot_chunk, dtype=tf.float32)
        target_slice_chunk = tf.convert_to_tensor(target_slice_chunk, dtype=tf.int32)
        pos_slice_chunk = tf.convert_to_tensor(pos_slice_chunk, dtype=tf.int32)

        if pos_mask is not None:
            pos_mask_chunk = tf.convert_to_tensor(pos_mask_chunk, dtype=tf.float32)

        # batching parameters
        num_batches = int(np.ceil(actual_chunk_size / batch_size))

        # loop over batches
        pred_batches = []
        for bi in range(num_batches):

            # collect batch
            seq_1hot_batch = seq_1hot_chunk[
                bi * batch_size : (bi + 1) * batch_size, ...
            ]
            target_slice_batch = target_slice_chunk[
                bi * batch_size : (bi + 1) * batch_size, ...
            ]
            pos_slice_batch = pos_slice_chunk[
                bi * batch_size : (bi + 1) * batch_size, ...
            ]

            pos_mask_batch = None
            if pos_mask is not None:
                pos_mask_batch = pos_mask_chunk[
                    bi * batch_size : (bi + 1) * batch_size, ...
                ]

            pred_batch = (
                _count_func(
                    model,
                    seq_1hot_batch,
                    target_slice_batch,
                    pos_slice_batch,
                    pos_mask_batch,
                    track_scale,
                    track_transform,
                    clip_soft,
                    pseudo_count,
                    untransform_old,
                    use_mean,
                )
                .numpy()
                .astype(dtype)
            )

            pred_batches.append(pred_batch)

        # concat predicted batches
        preds = np.concatenate(pred_batches, axis=0)

        pred_chunks.append(preds)

        # collect garbage
        gc.collect()

    # concat predicted chunks
    preds = np.concatenate(pred_chunks, axis=0)

    print("Made predictions in %ds" % (time.time() - t0))

    return preds


################################################################################
# __main__
# ###############################################################################
if __name__ == "__main__":
    main()
