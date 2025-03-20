#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pysam
import json

from Bio import SeqIO
from tqdm import tqdm
import tensorflow as tf
from baskerville import dna
from baskerville import dataset
from baskerville import seqnn

from utils import *


def load_model(params_file):
    """Load the model from parameters file."""
    with open(params_file) as params_open:
        params = json.load(params_open)

        params_model = params['model']
        params_train = params['train']
    
    return params_model, params_train


def initialize_models(model_file, params_model, target_index, n_folds=4):
    """Initialize models from file."""
    models = []
    for fold_ix in range(n_folds):
        model_file_path = os.path.join(model_file, f"f{str(fold_ix)}/model0_best.h5")

        if not os.path.exists(model_file_path):
            print(f"Model file {model_file_path} not found, skipping.")
            continue
        
        seqnn_model = seqnn.SeqNN(params_model)
        seqnn_model.restore(model_file_path, 0)
        seqnn_model.build_slice(target_index)
        seqnn_model.build_ensemble(True)
        
        models.append(seqnn_model)
    
    return models


def analyze_variant(vcf_file, fasta_file, model_file, params_file, targets_file, 
                   gencode_file, output_dir='output', 
                   tissue_names=None, fig_width=800):
    """Analyze a variant and generate plots."""
    if tissue_names is None:
        tissue_names = ['RNA:adipose_tissue',
                        'RNA:adrenal_gland',
                        'RNA:bladder',
                        'RNA:blood',
                        'RNA:blood_vessel',
                        'RNA:brain',
                        'RNA:breast',
                        'RNA:cervix_uteri',
                        'RNA:colon',
                        'RNA:esophagus',
                        'RNA:fallopian_tube',
                        'RNA:heart',
                        'RNA:kidney',
                        'RNA:liver',
                        'RNA:lung',
                        'RNA:muscle',
                        'RNA:nerve',
                        'RNA:ovary',
                        'RNA:pancreas',
                        'RNA:pituitary',
                        'RNA:prostate',
                        'RNA:salivary_gland',
                        'RNA:skin',
                        'RNA:small_intestine',
                        'RNA:spleen',
                        'RNA:stomach',
                        'RNA:testis',
                        'RNA:thyroid',
                        'RNA:uterus',
                        'RNA:vagina']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read parameters and load models
    params_model, _ = load_model(params_file)
    sequence_length = params_model['seq_length']

    # Read targets (GTEx RNA-seq)
    targets_df = pd.read_csv(targets_file, index_col=0, sep="\t")
    target_index = targets_df.index
    
    # Target selection
    targets_gtex = targets_df.iloc[np.where(['RNA:' in x for x in targets_df['description']])[0]]
    targets_gtex = targets_gtex.iloc[np.where(['GTEX-' in x for x in targets_gtex['identifier']])[0]]
    
    # Initialize model ensemble
    n_folds = 1
    models = initialize_models(model_file, params_model, target_index, n_folds)
    
    # Read genome
    fasta_open = pysam.Fastafile(fasta_file)
    
    # Read VCF
    vcf_df = pd.read_csv(vcf_file, header=None, index_col=None, sep='\t')
    
    # Extract variant information
    var_from = str(vcf_df.iloc[0][3])
    var_to = str(vcf_df.iloc[0][4])
    var_start = int(vcf_df.iloc[0][1])
    var_name = str(vcf_df.iloc[0][2])
    len_indel = abs(len(var_from) - len(var_to))
    center_pos = var_start - 1
    chrn = str(vcf_df.iloc[0][0])
    start = center_pos - sequence_length // 2
    end = center_pos + sequence_length // 2
    seq_len = end - start
    
    # Create temporary folder for intermediate files
    temp_folder = create_temp_folder()
    
    # Find bedtools path
    bedtools_exec = find_bedtools_path()
    
    # Create prediction BED file
    pred_bed_path = create_pred_bed(temp_folder, chrn, start, end)
    
    # Intersect with genes
    genes_intersect_path = intersect_with_genes(temp_folder, bedtools_exec, pred_bed_path, gencode_file)
    
    # Read intersected genes
    int_genes = pd.read_csv(genes_intersect_path, sep="\t", header=None)
    int_genes.columns = ['chrom','txStart','txEnd','name2','#name','strand','exonStarts','exonEnds','exonCount']
        
    int_genes['name_full'] = [f'{g} ({t})' for g,t in zip(int_genes['name2'], int_genes['#name'])]

    # Filter genes to plot
    #df_gene = int_genes.loc[int_genes.groupby('name2')['exonCount'].idxmax()]
    df_gene = sort_gene_annotations(int_genes)

    # Get transcript list
    list_transcripts = get_transcript_list(df_gene)
    
    # Prepare reference and alternate sequences
    ref_preds = []
    alt_preds = []
    
    # Process variant based on type (deletion or insertion)
    if var_from > var_to:
        # Deletion
        (seq_dna_left, seq_dna_right, seq_dna_alt) = parse_seqs_del(
            fasta_open, start, end, chrn, seq_len, len_indel, var_to, center_pos
        )
        sequence_one_hot_wt_left = dna.dna_1hot(seq_dna_left)
        sequence_one_hot_wt_right = dna.dna_1hot(seq_dna_right)
        sequence_one_hot_mut = dna.dna_1hot(seq_dna_alt)

        # Make predictions
        y_wt_left = np.mean(predict_tracks(models, sequence_one_hot_wt_left, n_folds)[0, :, :, :], axis=0)
        y_wt_right = np.mean(predict_tracks(models, sequence_one_hot_wt_right, n_folds)[0, :, :, :], axis=0)
        y_mut = np.mean(predict_tracks(models, sequence_one_hot_mut, n_folds)[0, :, :, :], axis=0)

        rpsf_left = dataset.untransform_preds1(y_wt_left, targets_df)
        rpsf_right = dataset.untransform_preds1(y_wt_right, targets_df)
        ref_preds.append(rpsf_left)
        ref_preds.append(rpsf_right)

        apsf = dataset.untransform_preds1(y_mut, targets_df)
        alt_preds.append(apsf)

        ref_preds = stitch_preds(ref_preds, [0])
        rp_snp = np.array(ref_preds)
        ap_snp = np.array(alt_preds)
    else:
        # Insertion
        (seq_dna, seq_dna_right, seq_dna_left) = parse_seqs_ins(
            fasta_open, start, end, chrn, seq_len, len_indel, var_to, center_pos
        )
        sequence_one_hot_wt = dna.dna_1hot(seq_dna)
        sequence_one_hot_mut_left = dna.dna_1hot(seq_dna_left)
        sequence_one_hot_mut_right = dna.dna_1hot(seq_dna_right)

        # Make predictions
        y_wt = np.mean(predict_tracks(models, sequence_one_hot_wt, n_folds)[0, :, :, :], axis=0)
        y_mut_left = np.mean(predict_tracks(models, sequence_one_hot_mut_left, n_folds)[0, :, :, :], axis=0)
        y_mut_right = np.mean(predict_tracks(models, sequence_one_hot_mut_right, n_folds)[0, :, :, :], axis=0)

        rpsf = dataset.untransform_preds1(y_wt, targets_df)
        ref_preds.append(rpsf)

        apsf_left = dataset.untransform_preds1(y_mut_left, targets_df)
        apsf_right = dataset.untransform_preds1(y_mut_right, targets_df)
        alt_preds.append(apsf_left)
        alt_preds.append(apsf_right)

        alt_preds = stitch_preds(alt_preds, [0])
        rp_snp = np.array(ref_preds)
        ap_snp = np.array(alt_preds)
    
    # Visualization parameters
    bins_full = 6144
    bins_toplot = bins_full
    variant_pos = bins_toplot // 2
    xlim = None
    
    # Prepare variant name for plotting
    if len(var_name) > 20:
        var_name = '_'.join(var_name.split('_')[0:2])
    
    # Determine sequence coordinates for plotting
    seq_out_st = center_pos - 32 * bins_toplot // 2
    seq_out_en = center_pos + 32 * bins_toplot // 2
    
    # Get transcript slices for plotting
    transcripts_slices = get_transcripts_slices(df_gene, list_transcripts, seq_out_st, seq_out_en, 
                                               var_from, var_to, len_indel, bins_toplot, xlim)
    
    # Plot coverage tracks for each tissue
    for tissue in tqdm(tissue_names):
        print(f"Processing {tissue}")
        targets_gtex_tissue = targets_gtex.iloc[np.where([x == tissue for x in targets_gtex['description']])[0]]
        index_targets = list(targets_gtex_tissue.index)
        
        ref_coverage, alt_coverage = get_sliced_coverage(rp_snp, ap_snp, bins_full, bins_toplot, index_targets)
        
        plot_coverage_tracks_plotly(ref_coverage, alt_coverage, variant_pos, f"{var_name}-{tissue.replace('/', '-').replace(':', '-')}", 
                                transcripts_slices, xlim=xlim, fig_height=None, fig_width=fig_width, savedir=output_dir)
            
    # Plot coverage tracks for all GTEx tissues combined
    index_targets = list(targets_gtex.index)
    ref_coverage, alt_coverage = get_sliced_coverage(rp_snp, ap_snp, bins_full, bins_toplot, index_targets)
    
    plot_coverage_tracks_plotly(ref_coverage, alt_coverage, variant_pos, f"{var_name}-GTEx-all", 
                            transcripts_slices, xlim=xlim, fig_height=None, fig_width=fig_width, savedir=output_dir)
        
    # Calculate coverage statistics for specific genes
    transcript_stats = {}
    for transcript_name in transcripts_slices.keys():
        index_targets = list(targets_gtex.index)
        
        ref_slice, alt_slice = get_transcript_coverage(transcripts_slices, transcript_name, rp_snp, ap_snp, index_targets)
        
        avg_ref = np.mean(ref_slice)
        avg_alt = np.mean(alt_slice)
        ratio = avg_alt / avg_ref
        
        transcript_stats[transcript_name] = {
            'ref_coverage': avg_ref,
            'alt_coverage': avg_alt,
            'alt/ref_ratio': ratio
        }
        
        print(f"Transcript {transcript_name}: ref_coverage={avg_ref:.2f}, alt_coverage={avg_alt:.2f}, alt/ref={ratio:.6f}")
    
    # Save stats to file
    stats_df = pd.DataFrame.from_dict(transcript_stats, orient='index')
    stats_df.to_csv(f"{output_dir}/{var_name}_transcript_stats.csv")
    
    print(f"Analysis complete. Results saved to {output_dir}")
    return transcript_stats


def main():
    parser = argparse.ArgumentParser(description="Analyze variant effect on gene expression using deep learning models")
    parser.add_argument("--vcf", required=True, help="VCF file containing the variant")
    parser.add_argument("--fasta", required=True, help="Reference genome FASTA file")
    parser.add_argument("--model", required=True, help="Model folder with folds")
    parser.add_argument("--params", required=True, help="JSON file with model parameters")
    parser.add_argument("--targets", required=True, help="Targets file")
    parser.add_argument("--gencode", required=True, help="Gencode BED file")
    parser.add_argument("--output_dir", default="output", help="Output directory")
    parser.add_argument("--tissues", nargs="+", default=None, help="List of tissues to analyze")
    parser.add_argument("--fig_width", type=int, default=800, help="Output figure width")
    
    args = parser.parse_args()

    # get working directory
    current_dir = os.getcwd()
    
    analyze_variant(
        vcf_file=args.vcf,
        fasta_file=os.path.join(current_dir, args.fasta),
        model_file=os.path.join(current_dir, args.model),
        params_file=os.path.join(current_dir, args.params),
        targets_file=os.path.join(current_dir, args.targets),
        gencode_file=os.path.join(current_dir, args.gencode),
        output_dir=os.path.join(current_dir, args.output_dir),
        tissue_names=args.tissues,
        fig_width=args.fig_width
    )


if __name__ == "__main__":
    main()