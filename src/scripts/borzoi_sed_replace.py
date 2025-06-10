#!/usr/bin/env python
# Copyright 2022 Calico LLC
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
from __future__ import print_function

from optparse import OptionParser
from collections import OrderedDict
import json
import pickle
import os
import pdb
import sys
import time
from tqdm import tqdm

import h5py
import numpy as np
import pandas as pd
import pybedtools
import pysam
from scipy.special import rel_entr
import tensorflow as tf

from baskerville.gene import Transcriptome
from baskerville import dataset
from baskerville import seqnn
from baskerville import vcf as bvcf
from baskerville import dna as dna_io

'''
borzoi_sed_replace.py

Compute Expression Difference (SED) scores for sequences in a tsv file,
relative to gene exons in a GTF file,
when substituting them in place of native genomic context specified in another tsv file.
'''

# helper function to get (potentially padded) sequence
def make_seq(genome_open, chrm, start, end, seq_len):
    if start < 0:
        seq_dna = "N" * (-start) + genome_open.fetch(chrm, 0, end)
    else:
        seq_dna = genome_open.fetch(chrm, start, end)

    # extend to full length
    if len(seq_dna) < seq_len:
        seq_dna += "N" * (seq_len - len(seq_dna))

    return seq_dna

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <model_file> <tsv_file>'
    parser = OptionParser(usage)
    parser.add_option(
        "-b",
        dest="bedgraph",
        default=False,
        action="store_true",
        help="Write ref/alt predictions as bedgraph [Default: %default]",
    )
    parser.add_option(
        "-f",
        dest="genome_fasta",
        default="%s/assembly/ucsc/hg38.fa" % os.environ.get('BORZOI_HG38', 'hg38'),
        help="Genome FASTA for sequences [Default: %default]",
    )
    parser.add_option(
        "-g",
        dest="genes_gtf",
        default="%s/genes/gencode41/gencode41_basic_nort.gtf" % os.environ.get('BORZOI_HG38', 'hg38'),
        help="GTF for gene definition [Default %default]",
    )
    parser.add_option(
        '--ctx',
        dest='ctx_tsv',
        default='/home/jlinder/seqnn/data/enhancers/unique_contexts_k562.tsv',
        help='TSV file containing genomic contexts to insert sequence into [Default %default]'
    )
    parser.add_option(
        "-o",
        dest="out_dir",
        default="sed",
        help="Output directory for tables and plots [Default: %default]",
    )
    parser.add_option(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Average forward and reverse complement predictions [Default: %default]",
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
        "--stats",
        dest="sed_stats",
        default="SED",
        help="Comma-separated list of stats to save. [Default: %default]",
    )
    parser.add_option(
        "-t",
        dest="targets_file",
        default=None,
        type="str",
        help="File specifying target indexes and labels in table format",
    )
    parser.add_option(
        "-u",
        dest="untransform_old",
        default=False,
        action="store_true",
    )
    parser.add_option(
        "--no_untransform",
        dest="no_untransform",
        default=False,
        action="store_true",
    )
    parser.add_option(
        '--no_unclip',
        dest='no_unclip',
        default=False,
        action='store_true'
    )
    parser.add_option(
        '-d',
        dest='data_head',
        default=None,
        type='int',
        help='Index for dataset/head [Default: %default]'
    )
    (options, args) = parser.parse_args()

    if len(args) == 3:
        # single worker
        params_file = args[0]
        model_file = args[1]
        tsv_file = args[2]

    elif len(args) == 4:
        # multi separate
        options_pkl_file = args[0]
        params_file = args[1]
        model_file = args[2]
        tsv_file = args[3]

        # save out dir
        out_dir = options.out_dir

        # load options
        options_pkl = open(options_pkl_file, 'rb')
        options = pickle.load(options_pkl)
        options_pkl.close()

        # update output directory
        options.out_dir = out_dir

    elif len(args) == 5:
        # multi worker
        options_pkl_file = args[0]
        params_file = args[1]
        model_file = args[2]
        tsv_file = args[3]
        worker_index = int(args[4])

        # load options
        options_pkl = open(options_pkl_file, 'rb')
        options = pickle.load(options_pkl)
        options_pkl.close()

        # update output directory
        options.out_dir = '%s/job%d' % (options.out_dir, worker_index)

    else:
        parser.error('Must provide parameters/model, tsv, and genes GTF')

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    options.shifts = [int(shift) for shift in options.shifts.split(',')]
    options.sed_stats = options.sed_stats.split(',')
    
    print("options.shifts = " + str(options.shifts))

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
    targets_strand_df = targets_prep_strand(targets_df)

    # set strand pairs (using new indexing)
    orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
    targets_strand_pair = np.array([orig_new_index[ti] for ti in targets_df.strand_pair])
    params_model['strand_pair'] = [targets_strand_pair]

    #################################################################
    # setup model

    seqnn_model = seqnn.SeqNN(params_model)
    if options.data_head is not None :
        print('data_head = ' + str(options.data_head), flush=True)
        seqnn_model.restore(model_file, options.data_head)
    else :
        seqnn_model.restore(model_file)
    seqnn_model.build_slice(targets_df.index)
    seqnn_model.build_ensemble(options.rc, [0])

    model_stride = seqnn_model.model_strides[0]
    out_seq_len = seqnn_model.target_lengths[0]*model_stride

    #################################################################
    # read sequence / contexts / genes

    # read sequences
    seq_df = pd.read_csv(tsv_file, sep='\t')[['id', 'seq']].copy().reset_index(drop=True)
    
    # read contexts
    ctx_df = pd.read_csv(options.ctx_tsv, sep='\t')[['chrom', 'start', 'end']].copy().reset_index(drop=True)
    ctx_df['midp'] = (ctx_df['start'] + ctx_df['end']) // 2
    ctx_df['ctx_id'] = ctx_df['chrom'].astype(str) + '_' + ctx_df['start'].astype(str) + '_' + ctx_df['end'].astype(str)

    # read genes
    transcriptome = Transcriptome(options.genes_gtf)
    gene_strand = {}
    for gene_id, gene in transcriptome.genes.items():
        gene_strand[gene_id] = gene.strand

    # map sequence contexts to gene positions
    ctx_gene_slice = map_ctx_genes(ctx_df, out_seq_len, transcriptome, model_stride, options.span, options.shifts[0])

    # remove contexts w/o genes
    num_ctxs_pre = len(ctx_df)
    ctx_gene_mask = np.array([len(cgs) > 0 for cgs in ctx_gene_slice])
    ctx_df = ctx_df.loc[ctx_gene_mask].copy().reset_index(drop=True)
    ctx_gene_slice = [ctx_gene_slice[si] for si in range(num_ctxs_pre) if ctx_gene_mask[si]]
    num_ctxs = len(ctx_df)

    #################################################################
    # setup output

    sed_out = initialize_output_h5(options.out_dir, options.sed_stats, seq_df, ctx_df, ctx_gene_slice, targets_strand_df)

    #################################################################
    # predict SNP scores, write output

    # create SNP seq generator
    genome_open = pysam.Fastafile(options.genome_fasta)

    # seq/context/gene index
    xi = 0

    # for each sequence
    for si, [_, seq_row] in tqdm(enumerate(seq_df.iterrows()), total=len(seq_df)):
        
        # get sequence
        insert_seq = seq_row['seq']
        insert_len = len(insert_seq)
        
        # for each context
        for ci, [_, ctx_row] in enumerate(ctx_df.iterrows()):
        
            # get context
            ctx_chrom = ctx_row['chrom']
            ctx_midp = ctx_row['midp']
            
            ctx_start = ctx_midp - seq_len // 2
            ctx_end = ctx_start + seq_len
            
            insert_start = ctx_midp - ctx_start - insert_len // 2
            insert_end = insert_start + insert_len
            
            if len(options.shifts) > 0 and options.shifts[0] != 0 :
                fetch_pad = 16384
                ctx_ref = make_seq(genome_open, ctx_chrom, ctx_start - fetch_pad, ctx_end + fetch_pad, seq_len + 2 * fetch_pad)
                ctx_alt = ctx_ref[:insert_start + fetch_pad] + insert_seq + ctx_ref[insert_end + fetch_pad:]
                ctx_ref = ctx_ref[fetch_pad+options.shifts[0]:fetch_pad+options.shifts[0]+seq_len]
                ctx_alt = ctx_alt[fetch_pad+options.shifts[0]:fetch_pad+options.shifts[0]+seq_len]
            else :
                ctx_ref = make_seq(genome_open, ctx_chrom, ctx_start, ctx_end, seq_len)
                ctx_alt = ctx_ref[:insert_start] + insert_seq + ctx_ref[insert_end:]
            
            ref_1hot = dna_io.dna_1hot(ctx_ref)
            alt_1hot = dna_io.dna_1hot(ctx_alt)
            
            seqs_1hot = np.concatenate([ref_1hot[None, ...], alt_1hot[None, ...]], axis=0)

            # get predictions
            if params_train['batch_size'] == 1:
                ref_preds = seqnn_model(seqs_1hot[:1])[0]
                alt_preds = seqnn_model(seqs_1hot[1:])[0]
            else:
                seq_preds = seqnn_model(seqs_1hot)
                ref_preds, alt_preds = seq_preds[0], seq_preds[1]

            # untransform predictions
            if options.targets_file is not None:
                if not options.no_untransform:
                    if options.untransform_old:
                        ref_preds = dataset.untransform_preds1(ref_preds, targets_df, unclip=not options.no_unclip)
                        alt_preds = dataset.untransform_preds1(alt_preds, targets_df, unclip=not options.no_unclip)
                    else:
                        ref_preds = dataset.untransform_preds(ref_preds, targets_df, unclip=not options.no_unclip)
                        alt_preds = dataset.untransform_preds(alt_preds, targets_df, unclip=not options.no_unclip)

            # for each overlapping gene
            for gene_id, gene_slice in ctx_gene_slice[ci].items():
                if len(gene_slice) > len(set(gene_slice)):
                    print('WARNING: %d %s has overlapping bins' % (ci,gene_id))

                # slice gene positions
                ref_preds_gene = ref_preds[gene_slice]
                alt_preds_gene = alt_preds[gene_slice]

                # slice relevant strand targets
                if gene_strand[gene_id] == '+':
                    gene_strand_mask = (targets_df.strand != '-')
                else:
                    gene_strand_mask = (targets_df.strand != '+')
                ref_preds_gene = ref_preds_gene[...,gene_strand_mask]
                alt_preds_gene = alt_preds_gene[...,gene_strand_mask]

                # compute pseudocounts
                ref_preds_strand = ref_preds[...,gene_strand_mask]
                pseudocounts = np.percentile(ref_preds_strand, 25, axis=0)

                # write scores to HDF
                write_snp(ref_preds_gene, alt_preds_gene, sed_out, xi, options.sed_stats, pseudocounts)

                xi += 1

    # close genome
    genome_open.close()

    ###################################################
    # compute SAD distributions across variants

    sed_out.close()


def clip_float(x, dtype=np.float16):
    return np.clip(x, np.finfo(dtype).min, np.finfo(dtype).max)


def initialize_output_h5(out_dir: str, sed_stats, seq_df, ctx_df, ctx_gene_slice, targets_df):
    """Initialize an output HDF5 file for SAD stats.
    
    Args:
            out_dir (str): Output directory.
            sed_stats (list): List of SAD stats to compute.
            seq_df (pandas.DataFrame): pandas dataframe with sequences.
            ctx_df (pandas.DataFrame): pandas dataframe with context coordinates.
            ctx_gene_slice ([dict]): List of dicts mapping gene_ids
                to their exon-overlapping positions for each sequence.
            targets_df (pandas.DataFrame): Targets table.
    """
    sed_out = h5py.File('%s/sed.h5' % out_dir, 'w')

    # collect identifier tuples
    seq_indexes = []
    ctx_indexes = []
    gene_ids = []
    snp_ids = []
    for seq_i, [_, row] in enumerate(seq_df.iterrows()):
        for ctx_i, gene_slice in enumerate(ctx_gene_slice):
            ctx_genes = list(gene_slice.keys())
            gene_ids += ctx_genes
            seq_indexes += [seq_i]*len(ctx_genes)
            ctx_indexes += [ctx_i]*len(ctx_genes)
    
    num_scores = len(seq_indexes)

    # write seq indexes
    seq_indexes = np.array(seq_indexes)
    sed_out.create_dataset('si', data=seq_indexes)
    
    # write ctx indexes
    ctx_indexes = np.array(ctx_indexes)
    sed_out.create_dataset('ci', data=ctx_indexes)

    # write genes
    gene_ids = np.array(gene_ids, 'S')
    sed_out.create_dataset('gene', data=gene_ids)

    # write seq ids
    seq_ids = np.array(seq_df['id'].values.tolist(), 'S')
    sed_out.create_dataset('seq_id', data=seq_ids)

    # write ctx ids
    ctx_ids = np.array(ctx_df['ctx_id'].values.tolist(), 'S')
    sed_out.create_dataset('ctx_id', data=ctx_ids)

    # write targets
    sed_out.create_dataset('target_ids', data=np.array(targets_df.identifier, 'S'))
    sed_out.create_dataset('target_labels', data=np.array(targets_df.description, 'S'))

    # initialize SED stats
    num_targets = targets_df.shape[0]
    for sed_stat in sed_stats:
        sed_out.create_dataset(sed_stat,
            shape=(num_scores, num_targets),
            dtype='float16')

    return sed_out


def make_ctx_bedt(ctx_df, seq_len: int, shift: int):
    """Make a BedTool object for all contexts, where seq_len considers cropping."""
    num_ctxs = len(ctx_df)
    left_len = seq_len // 2
    right_len = seq_len // 2
 
    ctx_bed_lines = []
    for si, [_, row] in enumerate(ctx_df.iterrows()):
        # bound sequence start at 0 (true sequence will be N padded)
        ctx_start = max(0, (row['midp'] + shift) - left_len)
        ctx_end = (row['midp'] + shift) + right_len
        ctx_bed_lines.append('%s %d %d %d' % (row['chrom'], ctx_start, ctx_end, si))

    ctx_bedt = pybedtools.BedTool('\n'.join(ctx_bed_lines), from_string=True)
    return ctx_bedt


def map_ctx_genes(
        ctx_df,
        seq_len: int,
        transcriptome,
        model_stride: int,
        span: bool,
        majority_overlap: bool=True,
        intron1: bool=False,
        shift: int=0):
    """Intersect contexts with gene exons, constructing a list
         mapping sequence indexes to dictionaries of gene_ids to their
         exon-overlapping positions in the sequence.
         
         Args:
                ctx_df (pandas.DataFrame): pandas dataframe with sequence context coordinates.
                seq_len (int): Sequence length, after model cropping.
                transcriptome (bgene.Transcriptome): Transcriptome.
                model_stride (int): Model stride.
                span (bool): If True, use gene span instead of exons.
                majority_overlap (bool): If True, only consider bins for which
                    the majority of the space overlaps an exon.
                intron1 (bool): If True, include intron bins adjacent to junctions.
         """

    # make gene BEDtool
    if span:
        genes_bedt = transcriptome.bedtool_span()
    else:
        genes_bedt = transcriptome.bedtool_exon()

    # make context BEDtool
    ctx_bedt = make_ctx_bedt(ctx_df, seq_len, shift)

    # map contexts to genes
    ctx_gene_slice = []
    for _, _ in ctx_df.iterrows():
        ctx_gene_slice.append(OrderedDict())

    for overlap in genes_bedt.intersect(ctx_bedt, wo=True):
        gene_id = overlap[3]
        gene_start = int(overlap[1])
        gene_end = int(overlap[2])
        seq_start = int(overlap[7])
        seq_end = int(overlap[8])
        si = int(overlap[9])

        # adjust for left overhang padded
        seq_len_chop = seq_end - seq_start
        seq_start -= (seq_len - seq_len_chop)

        # clip left boundaries
        gene_seq_start = max(0, gene_start - seq_start)
        gene_seq_end = max(0, gene_end - seq_start)

        if majority_overlap:
            # requires >50% overlap
            bin_start = int(np.round(gene_seq_start / model_stride))
            bin_end = int(np.round(gene_seq_end / model_stride))
        else:
            # any overlap
            bin_start = int(np.floor(gene_seq_start / model_stride))
            bin_end = int(np.ceil(gene_seq_end / model_stride))

        if intron1:
            bin_start -= 1
            bin_end += 1

        # clip boundaries
        bin_max = int(seq_len/model_stride)
        bin_start = min(bin_start, bin_max)
        bin_end = min(bin_end, bin_max)
        bin_start = max(0, bin_start)
        bin_end = max(0, bin_end)

        if bin_end - bin_start > 0:
            # save gene bin positions
            ctx_gene_slice[si].setdefault(gene_id,[]).extend(range(bin_start, bin_end))

    # handle possible overlaps
    for si in range(len(ctx_gene_slice)):
        for gene_id, gene_slice in ctx_gene_slice[si].items():
            ctx_gene_slice[si][gene_id] = np.unique(gene_slice)

    return ctx_gene_slice


def targets_prep_strand(targets_df):
    """Adjust targets table for merged stranded datasets."""
    # attach strand
    targets_strand = []
    for _, target in targets_df.iterrows():
        if target.strand_pair == target.name:
            targets_strand.append('.')
        else:
            targets_strand.append(target.identifier[-1])
    targets_df['strand'] = targets_strand

    # collapse stranded
    strand_mask = (targets_df.strand != '-')
    targets_strand_df = targets_df[strand_mask]

    return targets_strand_df


def write_pct(sed_out, sed_stats):
    """Compute percentile values for each target and write to HDF5."""
    # define percentiles
    d_fine = 0.001
    d_coarse = 0.01
    percentiles_neg = np.arange(d_fine, 0.1, d_fine)
    percentiles_base = np.arange(0.1, 0.9, d_coarse)
    percentiles_pos = np.arange(0.9, 1, d_fine)

    percentiles = np.concatenate([percentiles_neg, percentiles_base, percentiles_pos])
    sed_out.create_dataset('percentiles', data=percentiles)
    pct_len = len(percentiles)

    for sad_stat in sed_stats:
        if sad_stat not in ['REF','ALT']:
            sad_stat_pct = '%s_pct' % sad_stat

            # compute
            sad_pct = np.percentile(sed_out[sad_stat], 100*percentiles, axis=0).T
            sad_pct = sad_pct.astype('float16')

            # save
            sed_out.create_dataset(sad_stat_pct, data=sad_pct, dtype='float16')


def write_bedgraph_snp(snp, ref_preds, alt_preds, out_dir: str, model_stride: int):
    """Write full predictions around SNP as BedGraph.
    
    Args:
        snp (bvcf.SNP): SNP.
        ref_preds (np.ndarray): Reference predictions.
        alt_preds (np.ndarray): Alternate predictions.
        out_dir (str): Output directory.
        model_stride (int): Model stride.
    """
    target_length, num_targets = ref_preds.shape

    # mean across targets
    ref_preds = ref_preds.mean(axis=-1, dtype='float32')
    alt_preds = alt_preds.mean(axis=-1, dtype='float32')
    diff_preds = alt_preds - ref_preds

    # initialize raw predictions/targets
    ref_out = open('%s/%s_ref.bedgraph' % (out_dir, snp.rsid), 'w')
    alt_out = open('%s/%s_alt.bedgraph' % (out_dir, snp.rsid), 'w')
    diff_out = open('%s/%s_diff.bedgraph' % (out_dir, snp.rsid), 'w')

    # specify positions
    seq_len = target_length * model_stride
    left_len = seq_len // 2 - 1
    right_len = seq_len // 2
    seq_start = snp.pos - left_len - 1
    seq_end = snp.pos + right_len + max(0,
                                                                            len(snp.ref_allele) - snp.longest_alt())

    # write values
    bin_start = seq_start
    for bi in range(target_length):
        bin_end = bin_start + model_stride
        cols = [snp.chr, str(bin_start), str(bin_end), str(ref_preds[bi])]
        print('\t'.join(cols), file=ref_out)
        cols = [snp.chr, str(bin_start), str(bin_end), str(alt_preds[bi])]
        print('\t'.join(cols), file=alt_out)
        cols = [snp.chr, str(bin_start), str(bin_end), str(diff_preds[bi])]
        print('\t'.join(cols), file=diff_out)
        bin_start = bin_end

    ref_out.close()
    alt_out.close()
    diff_out.close()


def write_snp(ref_preds, alt_preds, sed_out, xi: int, sed_stats, pseudocounts):
    """Write SNP predictions to HDF, assuming the length dimension has
            been maintained.
            
        Args:
            ref_preds (np.ndarray): Reference predictions, (gene length x tasks)
            alt_preds (np.ndarray): Alternate predictions, (gene length x tasks)
            sed_out (h5py.File): HDF5 output file.
            xi (int): SNP index.
            sed_stats (list): SED statistics to compute.
            pseudocounts (np.ndarray): Target pseudocounts for safe logs.
        """

    # ref/alt_preds is L x T
    seq_len, num_targets = ref_preds.shape
    
    # log/sqrt
    ref_preds_log = np.log2(ref_preds+1)
    alt_preds_log = np.log2(alt_preds+1)

    # sum across length
    ref_preds_sum = ref_preds.sum(axis=0)
    alt_preds_sum = alt_preds.sum(axis=0)

    # difference of sums
    if 'SED' in sed_stats:
        sed = alt_preds_sum - ref_preds_sum
        sed_out['SED'][xi] = clip_float(sed).astype('float16')
    if 'logSED' in sed_stats:
        log_sed = np.log2(alt_preds_sum + 1) - np.log2(ref_preds_sum + 1)
        sed_out['logSED'][xi] = log_sed.astype('float16')

    # difference L1 norm
    if 'D1' in sed_stats:
        diff_abs = np.abs(ref_preds - alt_preds)
        diff_norm1 = diff_abs.sum(axis=0)
        sed_out['D1'][xi] = clip_float(diff_norm1).astype('float16')
    if 'logD1' in sed_stats:
        diff1_log = np.abs(ref_preds_log - alt_preds_log, 2)
        diff_log_norm1 = diff1_log.sum(axis=0)
        sed_out['logD1'][xi] = clip_float(diff_log_norm1).astype('float16')
    
    # difference L2 norm
    if 'D2' in sed_stats:
        diff2 = np.power(ref_preds - alt_preds, 2)
        diff_norm2 = np.sqrt(diff2.sum(axis=0))
        sed_out['D2'][xi] = clip_float(diff_norm2).astype('float16')
    if 'logD2' in sed_stats:
        diff2_log = np.power(ref_preds_log - alt_preds_log, 2)
        diff_log_norm2 = np.sqrt(diff2_log.sum(axis=0))
        sed_out['logD2'][xi] = clip_float(diff_log_norm2).astype('float16')

    # normalized scores
    ref_preds_norm = ref_preds + pseudocounts
    ref_preds_norm /= ref_preds_norm.sum(axis=0)
    alt_preds_norm = alt_preds + pseudocounts
    alt_preds_norm /= alt_preds_norm.sum(axis=0)

    # compare normalized squared difference
    if 'nD2' in sed_stats:
        ndiff2 = np.power(ref_preds_norm - alt_preds_norm, 2)
        ndiff_norm2 = np.sqrt(ndiff2.sum(axis=0))
        sed_out['nD2'][xi] = ndiff_norm2.astype('float16')

    # compare normalized abs max
    if 'nDi' in sed_stats:
        ndiff_abs = np.abs(ref_preds_norm - alt_preds_norm)
        ndiff_normi = ndiff_abs.max(axis=0)
        sed_out['nDi'][xi] = ndiff_normi.astype('float16')

    # compare normalized JS
    if 'JS' in sed_stats:
        ref_alt_entr = rel_entr(ref_preds_norm, alt_preds_norm).sum(axis=0)
        alt_ref_entr = rel_entr(alt_preds_norm, ref_preds_norm).sum(axis=0)
        js_dist = (ref_alt_entr + alt_ref_entr) / 2
        sed_out['JS'][xi] = js_dist.astype('float16')

    # predictions
    if 'REF' in sed_stats:
        sed_out['REF'][xi] = ref_preds_sum.astype('float16')
    if 'ALT' in sed_stats:
        sed_out['ALT'][xi] = alt_preds_sum.astype('float16')


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
