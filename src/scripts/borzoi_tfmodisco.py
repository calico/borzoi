#!/usr/bin/env python
from optparse import OptionParser
from collections import Counter
import os
import pdb
import subprocess
import time

import h5py
import numpy as np
import pandas as pd
import pybedtools
import pyranges
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from matplotlib import pyplot as plt
import seaborn as sns

from baskerville import dna_io
import pygene
import modisco

'''
borzoi_tfmodisco.py

Run TF Modisco on borzoi input saliency scores.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <scores_h5>'
    parser = OptionParser(usage)
    parser.add_option(
        '-c',
        dest='center_bp',
        default=None,
        type='int',
        help='Extract only center bp [Default: %default]',
    )
    parser.add_option(
        '-d',
        dest='meme_db',
        default='meme-5.4.1/motif_databases/CIS-BP_2.00/Homo_sapiens.meme',
        help='Meme database [Default: %default]',
    )
    parser.add_option(
        '-g',
        dest='genes_gtf_file',
        default='%s/genes/gencode38/gencode38_basic_protein.gtf' % os.environ.get('BORZOI_HG38', 'hg38'),
        help='Gencode GTF [Default: %default]',
    )
    parser.add_option(
        '--gc',
        dest='gc_content',
        default=0.41,
        type='float',
        help='Genome GC content [Default: %default]',
    )
    parser.add_option(
        '--fwd',
        dest='force_fwd',
        default=0,
        type='int',
        help='Do not use rev-comp in modisco [Default: %default]',
    )
    parser.add_option(
        '--modisco_window_size',
        dest='modisco_window_size',
        default=24,
        type='int',
        help='Modisco window size [Default: %default]',
    )
    parser.add_option(
        '--modisco_flank',
        dest='modisco_flank',
        default=8,
        type='int',
        help='Modisco flanks to add [Default: %default]',
    )
    parser.add_option(
        '--modisco_sliding_window_size',
        dest='modisco_sliding_window_size',
        default=18,
        type='int',
        help='Modisco sliding window size [Default: %default]',
    )
    parser.add_option(
        '--modisco_sliding_window_flank',
        dest='modisco_sliding_window_flank',
        default=8,
        type='int',
        help='Modisco sliding window flanks [Default: %default]',
    )
    parser.add_option(
        '--modisco_max_seqlets',
        dest='modisco_max_seqlets',
        default=20000,
        type='int',
        help='Modisco sliding window flanks [Default: %default]',
    )
    parser.add_option(
        '-i',
        dest='ic_t',
        default=0.1,
        type='float',
        help='Information content threshold [Default: %default]',
    )
    parser.add_option(
        '-n',
        dest='norm_type',
        default='max',
    )
    parser.add_option(
        '-o',
        dest='out_dir',
        default='tfm_out',
        help='Output directory [Default: %default]',
    )
    parser.add_option(
        '-r',
        dest='region',
        default=None,
        help='Limit to specific gene region [Default: %default',
    )
    parser.add_option(
        '-t',
        dest='targets_file',
        default=None,
        type='str',
        help='File specifying target indexes and labels in table format',
    )
    parser.add_option(
        '--kmer_len',
        dest='kmer_len',
        default=None,
        type='int',
        help='Extract only center bp [Default: %default]',
    )
    parser.add_option(
        '--num_gaps',
        dest='num_gaps',
        default=None,
        type='int',
        help='Extract only center bp [Default: %default]',
    )
    parser.add_option(
        '--num_mismatches',
        dest='num_mismatches',
        default=None,
        type='int',
        help='Extract only center bp [Default: %default]',
    )
    parser.add_option(
        '--clip_perc',
        dest='clip_perc',
        default=25,
        type='int',
        help='Percentile of max deviations to clip by [Default: %default]',
    )
    parser.add_option(
        '--tissue',
        dest='tissue',
        default=None,
        type='str',
        help='Main tissue name.',
    )
    parser.add_option(
        '--gene_file',
        dest='gene_file',
        default=None,
        type='str',
        help='Csv-file of gene metadata.',
    )
    
    (options,args) = parser.parse_args()

    if len(args) != 1:
        parser.error('Must provide Basenji nucleotide scores.')
    else:
        scores_h5_file = args[0]

    # setup output dir
    os.makedirs(options.out_dir, exist_ok=True)

    #Load gene dataframe and select tissue
    tissue_genes = None
    if options.gene_file is not None and options.tissue is not None :
        gene_df = pd.read_csv(options.gene_file, sep='\t')
        gene_df = gene_df.query("tissue == '" + str(options.tissue) + "'").copy().reset_index(drop=True)
        gene_df = gene_df.drop(columns=['Unnamed: 0'])

        print("len(gene_df) = " + str(len(gene_df)))

        #Get list of gene for tissue
        tissue_genes = gene_df['gene_base'].values.tolist()

        print("len(tissue_genes) = " + str(len(tissue_genes)))

    # read nucleotide scores
    t0 = time.time()
    print('Reading scores...', flush=True, end='')
    with h5py.File(scores_h5_file, 'r') as scores_h5:
        seq_len = scores_h5['grads'].shape[1]
        pos_start = seq_len//2 - options.center_bp//2
        pos_end = pos_start + options.center_bp
        hyp_scores = scores_h5['grads'][:,pos_start:pos_end]
        seqs_1hot = scores_h5['seqs'][:,pos_start:pos_end]
        seq_chrs = [chrm.decode('UTF-8') for chrm in scores_h5['chr']]
        seq_genes = [gene.decode('UTF-8') for gene in scores_h5['gene']]
        seq_starts = scores_h5['start'][:] + pos_start
        seq_ends = scores_h5['end'][:] - (seq_len - pos_end)
        
        if tissue_genes is not None :
            gene_dict = {gene.split(".")[0] : gene_i for gene_i, gene in enumerate(seq_genes)}

            #Get index of rows to keep
            keep_index = []
            for tissue_gene in tissue_genes :
                keep_index.append(gene_dict[tissue_gene])
            
            #Filter/sub-select data
            hyp_scores = hyp_scores[keep_index, ...]
            seqs_1hot = seqs_1hot[keep_index, ...]
            seq_chrs = [seq_chrs[k_ix] for k_ix in keep_index]
            seq_genes = [seq_genes[k_ix] for k_ix in keep_index]
            seq_starts = seq_starts[keep_index, ...]
            seq_ends = seq_ends[keep_index, ...]
            
            print("Filtered genes = " + str(hyp_scores.shape[0]))
        
    num_seqs, seq_len, _ = seqs_1hot.shape
    print('DONE in %ds.' % (time.time()-t0))

    # average across targets
    hyp_scores = hyp_scores.mean(axis=-1, dtype='float32')
    
    # normalize scores by sequence
    t0 = time.time()
    print('Normalizing scores...', flush=True, end='')
    if options.norm_type == 'max':
        scores_max = hyp_scores.std(axis=-1).max(axis=-1)
        max_clip = np.percentile(scores_max, options.clip_perc)
        scores_max = np.clip(scores_max, max_clip, np.inf)
        hyp_scores /= np.reshape(scores_max, (num_seqs,1,1))
    elif options.norm_type == 'gaussian':
        scores_std = hyp_scores.std(axis=-1)
        scores_std_wide = gaussian_filter1d(scores_std, sigma=1280, truncate=2)
        wide_clip = np.percentile(scores_std_wide, options.clip_perc)
        scores_std_wide = np.clip(scores_std_wide, wide_clip, np.inf)
        hyp_scores /= np.expand_dims(scores_std_wide, axis=-1)
    else:
        print('Unrecognized normalization %s' % options.norm_type)
    print('DONE in %ds.' % (time.time()-t0))

    ################################################
    # region filter

    if options.region is not None:
        hyp_scores, seqs_1hot = filter_region(hyp_scores, seqs_1hot,
            seq_genes, seq_starts, options.genes_gtf_file, options.region)

        # save to visualize individual examples
        with h5py.File('%s/scores.h5'%options.out_dir, 'w') as scores_h5:
            scores_h5.create_dataset('scores', data=hyp_scores, compression='gzip')
            scores_h5.create_dataset('seqs', data=seqs_1hot, compression='gzip')

    ################################################
    # tfmodisco

    if isinstance(hyp_scores, list):
        num_seqs = len(seqs_1hot)
        contrib_scores = [np.multiply(hyp_scores[si], seqs_1hot[si]) for si in range(num_seqs)]
    else:
        num_seqs = seqs_1hot.shape[0]
        contrib_scores = np.multiply(hyp_scores, seqs_1hot)

    # make seqlets to patterns factory
    if options.kmer_len is not None and options.num_gaps is not None and options.num_mismatches is not None :
        tfm_seqlets = modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
            trim_to_window_size=options.modisco_window_size,
            initial_flank_to_add=options.modisco_flank,
            kmer_len=options.kmer_len, num_gaps=options.num_gaps, num_mismatches=options.num_mismatches,
            final_min_cluster_size=20)
    else :
        tfm_seqlets = modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
            trim_to_window_size=options.modisco_window_size,
            initial_flank_to_add=options.modisco_flank,
            final_min_cluster_size=20)

    # make modisco workflow
    tfm_workflow = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
        sliding_window_size=options.modisco_sliding_window_size,
        flank_size=options.modisco_sliding_window_flank,
        max_seqlets_per_metacluster=options.modisco_max_seqlets,
        seqlets_to_patterns_factory=tfm_seqlets)

    # run modisco workflow
    task_label = 'out0'
    tfm_results = tfm_workflow(
     task_names=[task_label],
     contrib_scores={task_label: contrib_scores},
     hypothetical_contribs={task_label: hyp_scores},
     revcomp=False if options.force_fwd == 1 else True,
     one_hot=seqs_1hot)

    # save results
    tfm_h5_file = '%s/tfm.h5' % options.out_dir
    with h5py.File(tfm_h5_file, 'w') as tfm_h5:
        tfm_results.save_hdf5(tfm_h5)

    ################################################
    # extract motif PWMs

    at_pct = (1-options.gc_content)/2
    gc_pct = options.gc_content/2
    background = np.array([at_pct, gc_pct, gc_pct, at_pct])
    
    tfm_pwms = {}

    with h5py.File(tfm_h5_file, 'r') as tfm_h5:
        metacluster_names = [mcr.decode("utf-8") for mcr in list(tfm_h5["metaclustering_results"]["all_metacluster_names"][:])]
        for metacluster_name in metacluster_names:
            metacluster_grp = tfm_h5["metacluster_idx_to_submetacluster_results"][metacluster_name]
            all_patterns = metacluster_grp["seqlets_to_patterns_result"]["patterns"]["all_pattern_names"][:]
            all_pattern_names = [x.decode("utf-8") for x in list(all_patterns)]
            for pattern_name in all_pattern_names:
                pattern_id = (metacluster_name+'_'+pattern_name)
                pattern = metacluster_grp["seqlets_to_patterns_result"]["patterns"][pattern_name]
                fwd = np.array(pattern["sequence"]["fwd"])
                clip_pwm = ic_clip(fwd, options.ic_t, background)
                if clip_pwm is None: 
                    print('pattern_id: %s is skipped because no bp pass threshold.' % pattern_id)
                else:
                    tfm_pwms[pattern_id] = clip_pwm
                    print('pattern_id: %s is converted to pwm.' % pattern_id)

    ################################################
    # tomtom
                    
    # initialize MEME
    modisco_meme_file = options.out_dir+'/modisco_' + options.out_dir.replace("/", "_") + '.meme'
    modisco_meme_open = open(modisco_meme_file, 'w')

    # header
    modisco_meme_open.write('MEME version 4\n\n')
    modisco_meme_open.write('ALPHABET= ACGT\n\n')
    modisco_meme_open.write('strands: + -\n\n')
    modisco_meme_open.write('Background letter frequencies\n')
    modisco_meme_open.write('A %f C %f G %f T %f\n\n' % tuple(background))

    # PWMs
    for key in tfm_pwms.keys():
        modisco_meme_open.write('MOTIF '+key+'\n')
        modisco_meme_open.write('letter-probability matrix: alength= 4 w= ' + str(tfm_pwms[key].shape[0]) + '\n')
        np.savetxt(modisco_meme_open, tfm_pwms[key])
        modisco_meme_open.write('\n')

    modisco_meme_open.close()
                    
    # run tomtom
    tomtom_cmd = 'tomtom -dist pearson -thresh 0.1 -oc %s %s %s' % \
        (options.out_dir, modisco_meme_file, options.meme_db)
    subprocess.call(tomtom_cmd, shell=True)


def filter_region(scores, seqs_1hot, seq_genes, seq_starts, genes_gtf_file, region, min_size=64, ss_window=192, utr_window=192):
    """Filter scores and sequences for a specific gene region."""
    num_seqs, seq_len, _ = seqs_1hot.shape

    # parse GTF
    genes_gtf = pygene.GTF(genes_gtf_file)

    # collection regions
    scores_region = []
    seqs_1hot_region = []

    # for each gene sequence
    print('Extracting region %s...' % region)
    for gi in tqdm(range(num_seqs)):
        gene_id = seq_genes[gi]
        gene = genes_gtf.genes[gene_id]

        # collect regions
        region_starts = []
        region_ends = []
        for _, tx in gene.transcripts.items():
            tx.define_utrs()
            if region in ['3utr','utr3']:
                for utr in tx.utrs3:
                    region_starts.append(utr.start)
                    region_ends.append(utr.end)

            elif region.find('ss') != -1:
                if region in ['ss5', '5ss']:
                    if tx.strand == '+':
                        exon_side = 'end'
                    else:
                        exon_side = 'start'
                else:
                    if tx.strand == '+':
                        exon_side = 'start'
                    else:
                        exon_side = 'end'

                if exon_side == 'start':
                    for ei in range(1, len(tx.exons)):
                        ss_start = tx.exons[ei].start - ss_window//2
                        ss_end = ss_start + ss_window
                        region_starts.append(ss_start)
                        region_ends.append(ss_end)
                else:
                    for ei in range(len(tx.exons)-1):
                        ss_start = tx.exons[ei].end - ss_window//2
                        ss_end = ss_start + ss_window
                        region_starts.append(ss_start)
                        region_ends.append(ss_end)
            else:
                print('Unrecognized region %s' % region, file=sys.stderr)

        num_regions = len(region_starts)
        if num_regions > 0:
            # merge
            region_ranges = pyranges.PyRanges(chromosomes=[tx.chrom]*num_regions,
                starts=region_starts, ends=region_ends)
            region_ranges = region_ranges.merge()

            # for each region
            for _, rr in region_ranges.df.iterrows():
                # collect scores
                scores_start = max(0, rr.Start - seq_starts[gi])
                scores_end = min(seq_len, rr.End - seq_starts[gi])

                skip_region = False

                # check splice site length match
                if region.find('ss') != -1:
                    if scores_end - scores_start != ss_window:
                        skip_region = True
                
                else:
                    # sample variable length window
                    if scores_end - scores_start < utr_window:
                        skip_region = True
                    else:
                        scores_std = scores[gi,scores_start:scores_end].std(axis=-1)
                        scores_len = len(scores_std)
                        scores_peak = np.argmax(scores_std)
                        scores_peak = max(utr_window//2, scores_peak)
                        scores_peak = min(scores_len-utr_window//2, scores_peak)
                        scores_start += scores_peak - utr_window//2
                        scores_end = scores_start + utr_window

                if not skip_region:
                    scores_region_ri = scores[gi,scores_start:scores_end]
                    seqs_1hot_ri = seqs_1hot[gi,scores_start:scores_end]
                    if gene.strand == '-':
                        scores_region_ri = dna_io.hot1_rc(scores_region_ri)
                        seqs_1hot_ri = dna_io.hot1_rc(seqs_1hot_ri)
                    scores_region.append(scores_region_ri)
                    seqs_1hot_region.append(seqs_1hot_ri)

    scores_region = np.array(scores_region)
    seqs_1hot_region = np.array(seqs_1hot_region)

    return scores_region, seqs_1hot_region


def ic_clip(pwm, threshold, background=[0.25]*4):
    """Clip PWM sides with an information content threshold."""

    pc = 0.001
    odds_ratio = ((pwm+pc)/(1+4*pc)) / (background[None,:])
    ic = (np.log((pwm+pc)/(1+4*pc)) / np.log(2))*pwm
    ic -= (np.log(background)*background/np.log(2))[None,:]
    ic_total = np.sum(ic,axis=1)[:,None]
    
    # no bp pass threshold
    if ~np.any(ic_total.flatten()>threshold):
        return None
    else:
        left = np.where(ic_total>threshold)[0][0]
        right = np.where(ic_total>threshold)[0][-1]
        return pwm[left:(right+1)]

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
