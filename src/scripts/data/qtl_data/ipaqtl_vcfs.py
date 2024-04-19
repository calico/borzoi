#!/usr/bin/env python
from optparse import OptionParser
import os
import pdb
import time

import numpy as np
import pandas as pd
import pyranges as pr
from tqdm import tqdm

'''
ipaqtl_vcfs.py

Generate positive and negative intronic paQTL sets from the QTL catalog txrevise.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage)
    parser.add_option('--neg_pip', dest='neg_pip',
            default=0.01, type='float',
            help='PIP upper limit for negative examples. [Default: %default]')
    parser.add_option('--pos_pip', dest='pos_pip',
            default=0.9, type='float',
            help='PIP lower limit for positive examples. [Default: %default]')
    parser.add_option('--match_gene', dest='match_gene',
            default=0, type='int',
            help='Try finding negative in same gene as positive. [Default: %default]')
    parser.add_option('--match_allele', dest='match_allele',
            default=0, type='int',
            help='Try finding negative with same ref and alt alleles. [Default: %default]')
    parser.add_option('-o', dest='out_prefix',
            default='qtlcat_ipaqtl')
    (options,args) = parser.parse_args()

    tissue_name = options.out_prefix.split('txrev_')[1]
    
    gtf_file = '/home/drk/common/data/genomes/hg38/genes/gencode41/gencode41_basic_nort_protein.gtf'
    
    # read variant table
    qtlcat_df_neg = pd.read_csv("ge/GTEx_snps_" + tissue_name + "_intronic_polya_negatives.bed.gz", sep='\t')
    qtlcat_df_pos = pd.read_csv("txrev/GTEx_snps_" + tissue_name + "_intronic_polya_positives.bed.gz", sep='\t')
    
    # read TPM bin table and construct lookup dictionaries
    tpm_df = pd.read_csv('ge/GTEx_ge_' + tissue_name + "_tpms.csv", sep='\t')[['gene_id', 'tpm', 'bin_index', 'bin_index_l', 'bin_index_r']]
    gene_to_tpm_dict = tpm_df.set_index('gene_id').to_dict(orient='index')
    
    # filter on SNPs with genes in TPM bin dict
    qtlcat_df_neg = qtlcat_df_neg.loc[qtlcat_df_neg['gene_id'].isin(tpm_df['gene_id'].values.tolist())].copy().reset_index(drop=True)
    qtlcat_df_pos = qtlcat_df_pos.loc[qtlcat_df_pos['gene_id'].isin(tpm_df['gene_id'].values.tolist())].copy().reset_index(drop=True)
    
    #Load gene span annotation (protein-coding/categorized only)
    gtf_df = pd.read_csv(gtf_file, sep='\t', skiprows=5, names=['id_str'])
    gtf_genes = gtf_df['id_str'].apply(lambda x: x.split("gene_id \"")[1].split("\";")[0].split(".")[0]).unique().tolist()

    # filter on SNPs with genes in GTF file
    qtlcat_df_neg = qtlcat_df_neg.loc[qtlcat_df_neg['gene_id'].isin(gtf_genes)].copy().reset_index(drop=True)
    qtlcat_df_pos = qtlcat_df_pos.loc[qtlcat_df_pos['gene_id'].isin(gtf_genes)].copy().reset_index(drop=True)
    
    bin_to_genes_dict = {}
    for _, row in tpm_df.iterrows() :
        
        if row['bin_index'] not in bin_to_genes_dict :
            bin_to_genes_dict[row['bin_index']] = []
        
        bin_to_genes_dict[row['bin_index']].append(row['gene_id'])
    
    for sample_bin in bin_to_genes_dict :
        bin_to_genes_dict[sample_bin] = set(bin_to_genes_dict[sample_bin])

    # split molecular trait id and filter for polyadenylation (for positives)
    qtlcat_df_pos['gene'] = [mti.split('.')[0] for mti in qtlcat_df_pos.molecular_trait_id]
    qtlcat_df_pos['event'] = [mti.split('.')[2] for mti in qtlcat_df_pos.molecular_trait_id]

    qtlcat_df_pos = qtlcat_df_pos[qtlcat_df_pos.event == 'downstream']
    qtlcat_df_pos = qtlcat_df_pos.rename(columns={'distance' : 'pas_dist'})
    
    qtlcat_df_neg['molecular_trait_id'] = qtlcat_df_neg['gene_id'] + "." + "grp_0.downstream.negative"
    qtlcat_df_neg['gene'] = qtlcat_df_neg['gene_id']
    qtlcat_df_neg['event'] = 'downstream'
    qtlcat_df_neg = qtlcat_df_neg.rename(columns={'distance' : 'pas_dist'})

    paqtl_df = pd.concat([qtlcat_df_neg, qtlcat_df_pos]).copy().reset_index(drop=True)

    # determine positive variants
    paqtl_pos_df = paqtl_df[paqtl_df.pip >= options.pos_pip]
    paqtl_neg_df = paqtl_df[paqtl_df.pip < options.neg_pip]
    pos_variants = set(paqtl_pos_df.variant)
    
    neg_gene_and_allele_variants = 0
    neg_gene_variants = 0
    
    neg_expr_and_allele_variants = 0
    neg_expr_variants = 0
    
    unmatched_variants = 0

    # choose negative variants
    neg_variants = set()
    neg_dict = {}
    for pvariant in tqdm(pos_variants):
        paqtl_this_df = paqtl_pos_df[paqtl_pos_df.variant == pvariant]

        neg_found = False
        
        # optionally prefer negative from positive's gene set
        if options.match_gene == 1 and options.match_allele == 1 :
            pgenes = set(paqtl_this_df.gene)
            neg_found = find_negative(neg_variants, neg_dict, pos_variants, paqtl_this_df, paqtl_neg_df, pgenes, True)
            
            if neg_found :
                neg_gene_and_allele_variants += 1
        
        if not neg_found and options.match_gene == 1 :
            pgenes = set(paqtl_this_df.gene)
            neg_found = find_negative(neg_variants, neg_dict, pos_variants, paqtl_this_df, paqtl_neg_df, pgenes, False)
            
            if neg_found :
                neg_gene_variants += 1
        
        if not neg_found and options.match_allele == 1 :
            pgenes = bin_to_genes_dict[gene_to_tpm_dict[paqtl_this_df.iloc[0].gene]['bin_index']]
            neg_found = find_negative(neg_variants, neg_dict, pos_variants, paqtl_this_df, paqtl_neg_df, pgenes, True)
            
            if not neg_found and gene_to_tpm_dict[paqtl_this_df.iloc[0].gene]['bin_index'] != gene_to_tpm_dict[paqtl_this_df.iloc[0].gene]['bin_index_l'] :
                pgenes = bin_to_genes_dict[gene_to_tpm_dict[paqtl_this_df.iloc[0].gene]['bin_index_l']]
                neg_found = find_negative(neg_variants, neg_dict, pos_variants, paqtl_this_df, paqtl_neg_df, pgenes, True)
            
            if not neg_found and gene_to_tpm_dict[paqtl_this_df.iloc[0].gene]['bin_index'] != gene_to_tpm_dict[paqtl_this_df.iloc[0].gene]['bin_index_r'] :
                pgenes = bin_to_genes_dict[gene_to_tpm_dict[paqtl_this_df.iloc[0].gene]['bin_index_r']]
                neg_found = find_negative(neg_variants, neg_dict, pos_variants, paqtl_this_df, paqtl_neg_df, pgenes, True)
            
            if neg_found :
                neg_expr_and_allele_variants += 1
        
        if not neg_found :
            pgenes = bin_to_genes_dict[gene_to_tpm_dict[paqtl_this_df.iloc[0].gene]['bin_index']]
            neg_found = find_negative(neg_variants, neg_dict, pos_variants, paqtl_this_df, paqtl_neg_df, pgenes, False)
            
            if not neg_found and gene_to_tpm_dict[paqtl_this_df.iloc[0].gene]['bin_index'] != gene_to_tpm_dict[paqtl_this_df.iloc[0].gene]['bin_index_l'] :
                pgenes = bin_to_genes_dict[gene_to_tpm_dict[paqtl_this_df.iloc[0].gene]['bin_index_l']]
                neg_found = find_negative(neg_variants, neg_dict, pos_variants, paqtl_this_df, paqtl_neg_df, pgenes, False)
            
            if not neg_found and gene_to_tpm_dict[paqtl_this_df.iloc[0].gene]['bin_index'] != gene_to_tpm_dict[paqtl_this_df.iloc[0].gene]['bin_index_r'] :
                pgenes = bin_to_genes_dict[gene_to_tpm_dict[paqtl_this_df.iloc[0].gene]['bin_index_r']]
                neg_found = find_negative(neg_variants, neg_dict, pos_variants, paqtl_this_df, paqtl_neg_df, pgenes, False)
            
            if neg_found :
                neg_expr_variants += 1
        
        if not neg_found :
            print("[Warning] Could not find a matching negative for '" + pvariant + "'")
            unmatched_variants += 1

    print('%d positive variants' % len(pos_variants))
    print('%d negative variants' % len(neg_variants))
    print(' - %d gene-matched negatives with same alleles' % neg_gene_and_allele_variants)
    print(' - %d gene-matched negatives ' % neg_gene_variants)
    print(' - %d expr-matched negatives with same alleles' % neg_expr_and_allele_variants)
    print(' - %d expr-matched negatives ' % neg_expr_variants)
    print(' - %d unmatched negatives ' % unmatched_variants)

    pos_dict = {pv: pv for pv in pos_variants}
    
    # write VCFs
    write_vcf('%s_pos.vcf' % options.out_prefix, paqtl_df, pos_variants, pos_dict)
    write_vcf('%s_neg.vcf' % options.out_prefix, paqtl_df, neg_variants, neg_dict)

def find_negative(neg_variants, neg_dict, pos_variants, paqtl_this_df, paqtl_neg_df, pgenes, match_allele) :
    
    gene_mask = np.array([gene in pgenes for gene in paqtl_neg_df.gene])
    paqtl_neg_gene_df = paqtl_neg_df[gene_mask]

    # match PAS distance
    this_dist = paqtl_this_df.iloc[0].pas_dist
    dist_cmp = np.abs(paqtl_neg_gene_df.pas_dist - this_dist)
    dist_cmp_unique = np.sort(np.unique(dist_cmp.values))
    
    this_ref = paqtl_this_df.iloc[0].ref
    this_alt = paqtl_this_df.iloc[0].alt

    for ni_unique in dist_cmp_unique:
        
        paqtl_neg_gene_dist_df = paqtl_neg_gene_df.loc[dist_cmp == ni_unique]
        
        shuffle_index = np.arange(len(paqtl_neg_gene_dist_df), dtype='int32')
        np.random.shuffle(shuffle_index)
        
        for npaqtl_i in range(len(paqtl_neg_gene_dist_df)) :
            npaqtl = paqtl_neg_gene_dist_df.iloc[shuffle_index[npaqtl_i]]
        
            if not match_allele or (npaqtl.ref == this_ref and npaqtl.alt == this_alt):
                if npaqtl.variant not in neg_variants and npaqtl.variant not in pos_variants:

                    neg_variants.add(npaqtl.variant)
                    neg_dict[npaqtl.variant] = paqtl_this_df.iloc[0].variant

                    return True
    
    return False

def write_vcf(vcf_file, df, variants_write, variants_dict):
    vcf_open = open(vcf_file, 'w')
    print('##fileformat=VCFv4.2', file=vcf_open)
    print('##INFO=<ID=MT,Number=1,Type=String,Description="Molecular trait id">',
        file=vcf_open)
    print('##INFO=<ID=PD,Number=1,Type=Integer,Description="PAS distance">',
        file=vcf_open)
    print('##INFO=<ID=PI,Number=1,Type=String,Description="Positive SNP id">',
        file=vcf_open)
    cols = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO']
    print('\t'.join(cols), file=vcf_open)

    variants_written = set()

    for v in df.itertuples():
        if v.variant in variants_write and v.variant not in variants_written:
            cols = [v.chrom, str(v.pos), v.variant, v.ref, v.alt, '.', '.']
            cols += ['MT=%s;PD=%d;PI=%s' % (v.molecular_trait_id, v.pas_dist, variants_dict[v.variant])]
            print('\t'.join(cols), file=vcf_open)
            variants_written.add(v.variant)

    vcf_open.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
