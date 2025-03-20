#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pysam
import json
import glob

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


def score_STRs(input_folder, data_table, fasta_file, model_file, params_file, targets_file, 
                   gencode_file, output_dir='output'):
    
    tissue_keywords = {
        'Adipose_Subcutaneous': 'adipose',
        'Adipose_Visceral': 'adipose',
        'Adrenal_Gland': 'adrenal_gland',
        'Artery_Aorta': 'heart',
        'Artery_Tibial': 'heart',
        'Brain_Cerebellum': 'brain',
        'Brain_Cortex': 'brain',
        'Breast_Mammary_Tissue': 'breast',
        'Colon_Sigmoid': 'colon',
        'Colon_Transverse': 'colon',
        'Esophagus_Mucosa': 'esophagus',
        'Esophagus_Muscularis': 'esophagus',
        'Heart_LeftVentricle': 'heart',
        'Liver': 'liver',
        'Lung': 'lung',
        'Muscle_Skeletal': 'muscle',
        'Nerve_Tibial': 'nerve',
        'Ovary': 'ovary',
        'Pancreas': 'pancreas',
        'Pituitary': 'pituitary',
        'Prostate': 'prostate',
        'Skin_SunExposed': 'skin',
        'Skin_NotSunExposed': 'skin',
        'Spleen': 'spleen',
        'Stomach': 'stomach',
        'Testis': 'testis',
        'Thyroid': 'thyroid',
        'WholeBlood': 'blood',
        'Cells_Transformedfibroblasts': 'fibroblast'
    }

    # Create output directory
    if not os.path.exists(output_dir):
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
    n_folds = 4
    models = initialize_models(model_file, params_model, target_index, n_folds)
    
    # Create temporary folder for intermediate files
    temp_folder = create_temp_folder()
    
    # Find bedtools path
    bedtools_exec = find_bedtools_path()

    # Read genome
    fasta_open = pysam.Fastafile(fasta_file)

    df_repeats = pd.read_csv(data_table, index_col=0, header=0)
    df_repeats['max_tissue_trunc'] = ['_'.join(x.split('-')) for x in df_repeats['max_tissue']]

    for index, row in df_repeats.iterrows():

        vcf_name = row['vcf']
        tissue = row['max_tissue_trunc']
        gene_target = row['gene.name']

        if tissue not in tissue_keywords.keys():
            continue

        tissue_gtex = tissue_keywords[tissue]
        tissue_row_name = f'RNA:{tissue_gtex}'

        if tissue_gtex=='fibroblast':
            continue
        
        # find all files in the input folder that matches {vcf_name}_{n_repeat}.vcf
        pattern = f"{input_folder}/{vcf_name}_*.vcf"

        # List all files matching the pattern
        vcf_files = glob.glob(pattern)
                        
        genes_fc_gtex, fcs_gtex, repeats_gtex, abs_vals_ref_gtex, abs_vals_alt_gtex = [], [], [], [], []
        genes_fc_tissue, fcs_tissue, repeats_tissue, abs_vals_ref_tissue, abs_vals_alt_tissue = [], [], [], [], []
        first_pass = True
        
        for vcf_file in vcf_files:

            vcf_df = pd.read_csv(vcf_file, header=None, index_col=None, sep='\t')

            n_repeat = int(vcf_file.split('/')[-1].split('.')[0].split('_')[-1])

            var_from = str(vcf_df.iloc[0][3])
            var_to = str(vcf_df.iloc[0][4])
            var_start = int(vcf_df.iloc[0][1])
            var_name = str(vcf_df.iloc[0][2])
            len_indel = abs(len(var_from)-len(var_to))
            var_end = int(vcf_df.iloc[0][1]) + 1
            center_pos = var_start - 1
            chrn = str(vcf_df.iloc[0][0])
            start = center_pos - 524288 // 2
            end = center_pos + 524288 // 2
            seq_len = end - start

            if start<0:
                continue

            # Create prediction BED file
            pred_bed_path = create_pred_bed(temp_folder, chrn, start, end)
            
            # Intersect with genes
            genes_intersect_path = intersect_with_genes(temp_folder, bedtools_exec, pred_bed_path, gencode_file)
            
            # Read intersected genes
            int_genes = pd.read_csv(genes_intersect_path, sep="\t", header=None)
            int_genes.columns = ['chrom','txStart','txEnd','name_full','#name','strand','exonStarts','exonEnds','exonCount']
                
            # Filter genes to plot
            df_gene = int_genes.loc[int_genes.groupby('name_full')['exonCount'].idxmax()]

            # Get transcript list
            list_transcripts = get_transcript_list(df_gene)

            if first_pass:
                ref_preds = []
            alt_preds = []
            
            if var_from>var_to:
                
                # deletion
                (seq_dna_left, seq_dna_right, seq_dna_alt) = parse_seqs_del(
                    fasta_open, start, end, chrn, seq_len, len_indel, var_to, center_pos
                )
                sequence_one_hot_wt_left = dna.dna_1hot(seq_dna_left)
                sequence_one_hot_wt_right = dna.dna_1hot(seq_dna_right)
                sequence_one_hot_mut = dna.dna_1hot(seq_dna_alt)

                # Make predictions
                if first_pass:
                    # wt is the same, right shift is different
                    y_wt_left = np.mean(predict_tracks(models, sequence_one_hot_wt_left, n_folds)[0, :, :, :], axis=0)
                    rpsf_left = dataset.untransform_preds1(y_wt_left, targets_df)
                    first_pass = False

                y_wt_right = np.mean(predict_tracks(models, sequence_one_hot_wt_right, n_folds)[0, :, :, :], axis=0)
                rpsf_right = dataset.untransform_preds1(y_wt_right, targets_df)
                ref_preds.append(rpsf_left)
                ref_preds.append(rpsf_right)
                ref_preds = stitch_preds(ref_preds, [0])
                rp_snp = np.array(ref_preds)
                    
                y_mut = np.mean(predict_tracks(models, sequence_one_hot_mut, n_folds)[0, :, :, :], axis=0)

                apsf = dataset.untransform_preds1(y_mut, targets_df)
                alt_preds.append(apsf)
                ap_snp = np.array(alt_preds)

            else:
                
                # insertion
                (seq_dna, seq_dna_alt_right, seq_dna_alt_left) = parse_seqs_ins(
                    fasta_open, start, end, chrn, seq_len, len_indel, var_to, center_pos
                )
                sequence_one_hot_wt = dna.dna_1hot(seq_dna)
                sequence_one_hot_mut_left = dna.dna_1hot(seq_dna_alt_left)
                sequence_one_hot_mut_right = dna.dna_1hot(seq_dna_alt_right)

                # Make predictions
                if first_pass:
                    y_wt = np.mean(predict_tracks(models, sequence_one_hot_wt, n_folds)[0, :, :, :], axis=0)
                    rpsf = dataset.untransform_preds1(y_wt, targets_df)
                    ref_preds.append(rpsf)
                    rp_snp = np.array(ref_preds)
                    first_pass = False

                y_mut_left = np.mean(predict_tracks(models, sequence_one_hot_mut_left, n_folds)[0, :, :, :], axis=0)
                y_mut_right = np.mean(predict_tracks(models, sequence_one_hot_mut_right, n_folds)[0, :, :, :], axis=0)

                apsf_left = dataset.untransform_preds1(y_mut_left, targets_df)
                apsf_right = dataset.untransform_preds1(y_mut_right, targets_df)
                alt_preds.append(apsf_left)
                alt_preds.append(apsf_right)

                alt_preds = stitch_preds(alt_preds, [0])
                ap_snp = np.array(alt_preds)

            bins_full = 6144
            variant_pos = bins_full//2 

            seq_out_st = center_pos - 32*bins_full//2
            seq_out_en = center_pos + 32*bins_full//2

            transcripts_slices = get_transcripts_slices(df_gene, list_transcripts, seq_out_st, seq_out_en, 
                                                    var_from, var_to, len_indel, bins_full, None)
            
            var_name = '_'.join(var_name.split('_')[0:2])

            # ============================
            # save for specific tissue
            
            targets_gtex_tissue = targets_gtex.iloc[np.where([x==tissue_row_name for x in targets_gtex['description']])[0]]

            index_targets = list(targets_gtex_tissue.index)

            ref_coverage = np.mean(rp_snp[0,:,index_targets], axis=0)
            alt_coverage = np.mean(ap_snp[0,:,index_targets], axis=0)

            for gene in transcripts_slices.keys():
                exons = transcripts_slices[gene]
                # make arrays of indices from slices for each transcript
                ind_toslice = []
                for exon in exons:
                    ind_toslice.extend(list(range(exon[0], exon[1]+1)))
                x_cov = ref_coverage[ind_toslice].sum()
                y_cov = alt_coverage[ind_toslice].sum()
                abs_vals_ref_tissue.append(x_cov)
                abs_vals_alt_tissue.append(y_cov)
                genes_fc_tissue.append(gene[:-1])
                fcs_tissue.append(y_cov/x_cov)
                repeats_tissue.append(n_repeat)

            df_fc_tissue = pd.DataFrame({'gene': genes_fc_tissue, 
                                'fc': fcs_tissue, 
                                'repeat': repeats_tissue,
                                'abs_ref': abs_vals_ref_tissue,
                                'abs_alt': abs_vals_alt_tissue})
            
            df_fc_tissue.to_csv(f'{output_dir}/{vcf_name}_{tissue_gtex}.csv', index=False)

            # ============================
            # save for all GTEx tissues

            index_targets = list(targets_gtex.index)

            ref_coverage = np.mean(rp_snp[0,:,index_targets], axis=0)
            alt_coverage = np.mean(ap_snp[0,:,index_targets], axis=0)

            for gene in transcripts_slices.keys():
                exons = transcripts_slices[gene]
                # make arrays of indices from slices for each transcript
                ind_toslice = []
                for exon in exons:
                    ind_toslice.extend(list(range(exon[0], exon[1]+1)))
                x_cov = ref_coverage[ind_toslice].sum()
                y_cov = alt_coverage[ind_toslice].sum()
                abs_vals_ref_gtex.append(x_cov)
                abs_vals_alt_gtex.append(y_cov)
                genes_fc_gtex.append(gene[:-1])
                fcs_gtex.append(y_cov/x_cov)
                repeats_gtex.append(n_repeat)

            df_fc_gtex = pd.DataFrame({'gene': genes_fc_gtex, 
                                'fc': fcs_gtex, 
                                'repeat': repeats_gtex,
                                'abs_ref': abs_vals_ref_gtex,
                                'abs_alt': abs_vals_alt_gtex})
            
            df_fc_gtex.to_csv(f'{output_dir}/{vcf_name}_GTEx.csv', index=False)



def main():
    parser = argparse.ArgumentParser(description="Analyze variant effect on gene expression using deep learning models")
    parser.add_argument("--table", required=True, help="original STR data table with tissue information")
    parser.add_argument("--input", required=True, help="folder with STR .vcf files")
    parser.add_argument("--fasta", required=True, help="Reference genome FASTA file")
    parser.add_argument("--model", required=True, help="Model folder with folds")
    parser.add_argument("--params", required=True, help="JSON file with model parameters")
    parser.add_argument("--targets", required=True, help="Targets file")
    parser.add_argument("--gencode", required=True, help="Gencode BED file")
    parser.add_argument("--output_dir", default="output", help="Output directory")

    args = parser.parse_args()

    # get working directory
    current_dir = os.getcwd()
    
    score_STRs(
        input_folder=os.path.join(current_dir, args.input),
        data_table=os.path.join(current_dir, args.table),
        fasta_file=os.path.join(current_dir, args.fasta),
        model_file=os.path.join(current_dir, args.model),
        params_file=os.path.join(current_dir, args.params),
        targets_file=os.path.join(current_dir, args.targets),
        gencode_file=os.path.join(current_dir, args.gencode),
        output_dir=os.path.join(current_dir, args.output_dir),
    )


if __name__ == "__main__":
    main()