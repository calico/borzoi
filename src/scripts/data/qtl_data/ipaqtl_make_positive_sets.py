#!/usr/bin/env python
from optparse import OptionParser

import os

import util

import numpy as np
import pandas as pd

import pyranges as pr

'''
paqtl_make_positive_sets.py

Build tables with positive (causal) SNPs for paQTLs.
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
    apa_file = 'polyadb_intron.bed'
    gtf_file = '/home/drk/common/data/genomes/hg38/genes/gencode41/gencode41_basic_nort.gtf'

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
    
    #Compile positive SNP set for each tissue
    for tissue_name in tissue_names :
        
        print("-- " + str(tissue_name) + " --")

        #Load fine-mapping table
        vcf_df = pd.read_csv("txrev/GTEx_txrev_" + tissue_name + ".purity_filtered.txt.gz", sep='\t', usecols=['chromosome', 'position', 'ref', 'alt', 'variant', 'pip', 'molecular_trait_id'], low_memory=False)

        #Only keep SNPs (no indels)
        vcf_df = vcf_df.loc[(vcf_df['ref'].str.len() == vcf_df['alt'].str.len()) & (vcf_df['ref'].str.len() == 1)].copy().reset_index(drop=True)
        
        #Only keep SNPs associated with polyadenylation events
        vcf_df = vcf_df.loc[vcf_df['molecular_trait_id'].str.contains(".downstream.")].copy().reset_index(drop=True)

        vcf_df['chromosome'] = 'chr' + vcf_df['chromosome'].astype(str)
        vcf_df['start'] = vcf_df['position'].astype(int)
        vcf_df['end'] = vcf_df['start'] + 1
        vcf_df['strand'] = "."

        vcf_df = vcf_df[['chromosome', 'start', 'end', 'ref', 'alt', 'strand', 'variant', 'pip', 'molecular_trait_id']]
        vcf_df = vcf_df.rename(columns={'chromosome' : 'Chromosome', 'start' : 'Start', 'end' : 'End', 'strand' : 'Strand'})

        print("len(vcf_df) = " + str(len(vcf_df)))

        #Load polyadenylation site annotation
        apa_df = pd.read_csv(apa_file, sep='\t', names=['Chromosome', 'Start', 'End', 'pas_id', 'feat1', 'Strand'])
        apa_df['Start'] += 1
        
        #Load gene span annotation
        gtf_df = pd.read_csv(gtf_file, sep='\t', skiprows=5, names=['Chromosome', 'havana_str', 'feature', 'Start', 'End', 'feat1', 'Strand', 'feat2', 'id_str'])
        gtf_df = gtf_df.query("feature == 'gene'").copy().reset_index(drop=True)

        gtf_df['gene_id'] = gtf_df['id_str'].apply(lambda x: x.split("gene_id \"")[1].split("\";")[0].split(".")[0])

        gtf_df = gtf_df[['Chromosome', 'Start', 'End', 'gene_id', 'feat1', 'Strand']].drop_duplicates(subset=['gene_id'], keep='first').copy().reset_index(drop=True)

        gtf_df['Start'] = gtf_df['Start'].astype(int) - gene_pad
        gtf_df['End'] = gtf_df['End'].astype(int) + gene_pad
        
        #Join dataframes against gtf annotation
        apa_pr = pr.PyRanges(apa_df)
        gtf_pr = pr.PyRanges(gtf_df)
        vcf_pr = pr.PyRanges(vcf_df)

        apa_gtf_pr = apa_pr.join(gtf_pr, strandedness='same')
        vcf_gtf_pr = vcf_pr.join(gtf_pr, strandedness=False)

        apa_gtf_df = apa_gtf_pr.df[['Chromosome', 'Start', 'End', 'pas_id', 'gene_id', 'Strand']].copy().reset_index(drop=True)
        vcf_gtf_df = vcf_gtf_pr.df[['Chromosome', 'Start', 'End', 'ref', 'alt', 'Strand', 'gene_id', 'variant', 'pip', 'molecular_trait_id']].copy().reset_index(drop=True)

        apa_gtf_df['Start'] -= max_distance
        apa_gtf_df['End'] += max_distance
        
        #Join vcf against polyadenylation annotation
        apa_gtf_pr = pr.PyRanges(apa_gtf_df)
        vcf_gtf_pr = pr.PyRanges(vcf_gtf_df)

        vcf_apa_pr = vcf_gtf_pr.join(apa_gtf_pr, strandedness=False)
        
        #Force gene_id of SNP to be same as the gene_id of the polyA site        
        vcf_apa_df = vcf_apa_pr.df.query("gene_id == gene_id_b").copy().reset_index(drop=True)
        vcf_apa_df = vcf_apa_df[['Chromosome', 'Start', 'ref', 'alt', 'gene_id', 'pas_id', 'Strand_b', 'Start_b', 'variant', 'pip', 'molecular_trait_id']]
        
        #Force gene_id of SNP to be same as the gene_id of the finemapped molecular trait
        vcf_apa_df['molecular_trait_gene_id'] = vcf_apa_df['molecular_trait_id'].apply(lambda x: x.split(".")[0])
        vcf_apa_df = vcf_apa_df.query("gene_id == molecular_trait_gene_id").copy().reset_index(drop=True)

        #PolyA site position
        vcf_apa_df['Start_b'] += max_distance
        vcf_apa_df = vcf_apa_df.rename(columns={'Start' : 'Pos', 'Start_b' : 'pas_pos', 'Strand_b' : 'Strand'})

        #Distance to polyA site
        vcf_apa_df['distance'] = np.abs(vcf_apa_df['Pos'] - vcf_apa_df['pas_pos'])
        
        #Choose unique SNPs by shortest distance to polyA site (and inverse PIP for tie-breaking)
        vcf_apa_df['pip_inv'] = 1. - vcf_apa_df['pip']
        
        vcf_apa_df = vcf_apa_df.sort_values(by=['distance', 'pip_inv'], ascending=True).drop_duplicates(subset=['Chromosome', 'Pos', 'ref', 'alt'], keep='first').copy().reset_index(drop=True)
        vcf_apa_df = vcf_apa_df.sort_values(['Chromosome', 'Pos', 'alt'], ascending=True).copy().reset_index(drop=True)

        vcf_df_filtered = vcf_apa_df.rename(columns={'Chromosome' : 'chrom', 'Pos' : 'pos', 'Strand' : 'strand'})
        vcf_df_filtered = vcf_df_filtered[['chrom', 'pos', 'ref', 'alt', 'gene_id', 'pas_id', 'strand', 'pas_pos', 'distance', 'variant', 'pip', 'molecular_trait_id']]

        print("len(vcf_df_filtered) = " + str(len(vcf_df_filtered)))
        
        #Store intermediate SNPs (filtered)
        vcf_df_filtered.to_csv("txrev/GTEx_snps_" + tissue_name + "_intronic_polya_finemapped_filtered.bed.gz", sep='\t', index=False)
        
        #Reload filtered SNP file
        vcf_df_filtered = pd.read_csv("txrev/GTEx_snps_" + tissue_name + "_intronic_polya_finemapped_filtered.bed.gz", sep='\t', compression='gzip')

        #Only keep SNPs with PIP > cutoff
        pos_df = vcf_df_filtered.query("pip > " + str(pip_cutoff)).copy().reset_index(drop=True)

        #Store final table of positive SNPs
        pos_df.to_csv("txrev/GTEx_snps_" + tissue_name + "_intronic_polya_positives.bed.gz", sep='\t', index=False)
        
        print("len(pos_df) = " + str(len(pos_df)))

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
