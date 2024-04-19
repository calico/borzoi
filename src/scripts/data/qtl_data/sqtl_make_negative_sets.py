#!/usr/bin/env python
from optparse import OptionParser

import os

import util

import numpy as np
import pandas as pd

import pyranges as pr

'''
sqtl_make_negative_sets.py

Build tables with negative (non-causal) SNPs for sQTLs.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    #parser.add_option()
    (options,args) = parser.parse_args()
    
    #Parameters
    pip_cutoff = 0.01
    max_distance = 10000
    gene_pad = 50
    splice_file = '/home/drk/common/data/genomes/hg38/genes/gencode41/gencode41_basic_protein_splice.gff'
    gtf_file = '/home/drk/common/data/genomes/hg38/genes/gencode41/gencode41_basic_nort.gtf'
    finemap_file = 'txrev/GTEx_txrev_finemapped_merged.csv.gz'

    #Define tissues
    tissue_names = [
        'adipose_subcutaneous',
        'adipose_visceral',
        'adrenal_gland',
        'artery_aorta',
        'artery_coronary',
        'artery_tibial',
        'blood',
        'brain_amygdala',
        'brain_anterior_cingulate_cortex',
        'brain_caudate',
        'brain_cerebellar_hemisphere',
        'brain_cerebellum',
        'brain_cortex',
        'brain_frontal_cortex',
        'brain_hippocampus',
        'brain_hypothalamus',
        'brain_nucleus_accumbens',
        'brain_putamen',
        'brain_spinal_cord',
        'brain_substantia_nigra',
        'breast',
        'colon_sigmoid',
        'colon_transverse',
        'esophagus_gej',
        'esophagus_mucosa',
        'esophagus_muscularis',
        'fibroblast',
        'heart_atrial_appendage',
        'heart_left_ventricle',
        'kidney_cortex',
        'LCL',
        'liver',
        'lung',
        'minor_salivary_gland',
        'muscle',
        'nerve_tibial',
        'ovary',
        'pancreas',
        'pituitary',
        'prostate',
        'skin_not_sun_exposed',
        'skin_sun_exposed',
        'small_intestine',
        'spleen',
        'stomach',
        'testis',
        'thyroid',
        'uterus',
        'vagina',
    ]
    
    #Compile negative SNP set for each tissue
    for tissue_name in tissue_names :
        
        print("-- " + str(tissue_name) + " --")

        #Load summary stats and extract unique set of SNPs
        vcf_df = pd.read_csv("ge/GTEx_ge_" + tissue_name + ".all.tsv.gz", sep='\t', compression='gzip', usecols=['chromosome', 'position', 'ref', 'alt']).drop_duplicates(subset=['chromosome', 'position', 'ref', 'alt'], keep='first').copy().reset_index(drop=True)

        #Only keep SNPs (no indels)
        vcf_df = vcf_df.loc[(vcf_df['ref'].str.len() == vcf_df['alt'].str.len()) & (vcf_df['ref'].str.len() == 1)].copy().reset_index(drop=True)

        vcf_df['chromosome'] = 'chr' + vcf_df['chromosome'].astype(str)
        vcf_df['start'] = vcf_df['position'].astype(int)
        vcf_df['end'] = vcf_df['start'] + 1
        vcf_df['strand'] = "."

        vcf_df = vcf_df[['chromosome', 'start', 'end', 'ref', 'alt', 'strand']]
        vcf_df = vcf_df.rename(columns={'chromosome' : 'Chromosome', 'start' : 'Start', 'end' : 'End', 'strand' : 'Strand'})

        print("len(vcf_df) = " + str(len(vcf_df)))

        #Store intermediate SNPs
        #vcf_df.to_csv("ge/GTEx_snps_" + tissue_name + ".bed.gz", sep='\t', index=False, header=False)
        
        #Load splice site annotation
        splice_df = pd.read_csv(splice_file, sep='\t', names=['Chromosome', 'havana_str', 'feature', 'Start', 'End', 'feat1', 'Strand', 'feat2', 'id_str'], usecols=['Chromosome', 'Start', 'End', 'feature', 'feat1', 'Strand'])[['Chromosome', 'Start', 'End', 'feature', 'feat1', 'Strand']]
        
        #Load gene span annotation
        gtf_df = pd.read_csv(gtf_file, sep='\t', skiprows=5, names=['Chromosome', 'havana_str', 'feature', 'Start', 'End', 'feat1', 'Strand', 'feat2', 'id_str'])
        gtf_df = gtf_df.query("feature == 'gene'").copy().reset_index(drop=True)

        gtf_df['gene_id'] = gtf_df['id_str'].apply(lambda x: x.split("gene_id \"")[1].split("\";")[0].split(".")[0])

        gtf_df = gtf_df[['Chromosome', 'Start', 'End', 'gene_id', 'feat1', 'Strand']].drop_duplicates(subset=['gene_id'], keep='first').copy().reset_index(drop=True)

        gtf_df['Start'] = gtf_df['Start'].astype(int) - gene_pad
        gtf_df['End'] = gtf_df['End'].astype(int) + gene_pad
        
        #Join dataframes against gtf annotation
        splice_pr = pr.PyRanges(splice_df)
        gtf_pr = pr.PyRanges(gtf_df)
        vcf_pr = pr.PyRanges(vcf_df)

        splice_gtf_pr = splice_pr.join(gtf_pr, strandedness='same')
        vcf_gtf_pr = vcf_pr.join(gtf_pr, strandedness=False)

        splice_gtf_df = splice_gtf_pr.df[['Chromosome', 'Start', 'End', 'feature', 'gene_id', 'Strand']].copy().reset_index(drop=True)
        vcf_gtf_df = vcf_gtf_pr.df[['Chromosome', 'Start', 'End', 'ref', 'alt', 'Strand', 'gene_id']].copy().reset_index(drop=True)

        splice_gtf_df['Start'] -= max_distance
        splice_gtf_df['End'] += max_distance
        
        #Join vcf against splice annotation
        splice_gtf_pr = pr.PyRanges(splice_gtf_df)
        vcf_gtf_pr = pr.PyRanges(vcf_gtf_df)

        vcf_splice_pr = vcf_gtf_pr.join(splice_gtf_pr, strandedness=False)
        
        #Force gene_id of SNP to be same as the gene_id of the splice site        
        vcf_splice_df = vcf_splice_pr.df.query("gene_id == gene_id_b").copy().reset_index(drop=True)
        vcf_splice_df = vcf_splice_df[['Chromosome', 'Start', 'ref', 'alt', 'gene_id', 'feature', 'Strand_b', 'Start_b']]

        #Splice site position
        vcf_splice_df['Start_b'] += max_distance
        vcf_splice_df = vcf_splice_df.rename(columns={'Start' : 'Pos', 'Start_b' : 'splice_pos', 'Strand_b' : 'Strand'})

        #Distance to splice site
        vcf_splice_df['distance'] = np.abs(vcf_splice_df['Pos'] - vcf_splice_df['splice_pos'])
        
        #Choose unique SNPs by shortest distance to splice site
        vcf_splice_df = vcf_splice_df.sort_values(by='distance', ascending=True).drop_duplicates(subset=['Chromosome', 'Pos', 'ref', 'alt'], keep='first').copy().reset_index(drop=True)
        vcf_splice_df = vcf_splice_df.sort_values(['Chromosome', 'Pos', 'alt'], ascending=True).copy().reset_index(drop=True)

        vcf_df_filtered = vcf_splice_df.rename(columns={'Chromosome' : 'chrom', 'Pos' : 'pos', 'Strand' : 'strand'})
        vcf_df_filtered = vcf_df_filtered[['chrom', 'pos', 'ref', 'alt', 'gene_id', 'feature', 'strand', 'splice_pos', 'distance']]

        print("len(vcf_df_filtered) = " + str(len(vcf_df_filtered)))
        
        #Store intermediate SNPs (filtered)
        vcf_df_filtered.to_csv("ge/GTEx_snps_" + tissue_name + "_splice_filtered.bed.gz", sep='\t', index=False)
        
        #Reload filtered SNP file
        vcf_df_filtered = pd.read_csv("ge/GTEx_snps_" + tissue_name + "_splice_filtered.bed.gz", sep='\t', compression='gzip')

        #Create variant identifier
        vcf_df_filtered['variant'] = vcf_df_filtered['chrom'] + "_" + vcf_df_filtered['pos'].astype(str) + "_" + vcf_df_filtered['ref'] + "_" + vcf_df_filtered['alt']
        
        #Load merged fine-mapping dataframe
        finemap_df = pd.read_csv(finemap_file, sep='\t')[['variant', 'pip']]
        
        #Join against fine-mapping dataframe
        neg_df = vcf_df_filtered.join(finemap_df.set_index('variant'), on='variant', how='left')
        neg_df.loc[neg_df['pip'].isnull(), 'pip'] = 0.
        
        #Only keep SNPs with PIP < cutoff
        neg_df = neg_df.query("pip < " + str(pip_cutoff)).copy().reset_index(drop=True)

        #Store final table of negative SNPs
        neg_df.to_csv("ge/GTEx_snps_" + tissue_name + "_splice_negatives.bed.gz", sep='\t', index=False)
        
        print("len(neg_df) = " + str(len(neg_df)))

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
