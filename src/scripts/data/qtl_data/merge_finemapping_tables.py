#!/usr/bin/env python
from optparse import OptionParser

import os

import util

import numpy as np
import pandas as pd

'''
merge_finemapping_tables.py

Merge fine-mapping tables of QTL credible sets.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    #parser.add_option()
    (options,args) = parser.parse_args()

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
    
    #Load and merge fine-mapping results
    dfs = []
    for tissue_name in tissue_names :

        print("-- " + tissue_name + " --")

        df = pd.read_csv("txrev/GTEx_txrev_" + tissue_name + ".purity_filtered.txt.gz", sep='\t', usecols=['chromosome', 'position', 'ref', 'alt', 'variant', 'pip'], low_memory=False)
        dfs.append(df.sort_values(by='pip', ascending=False).drop_duplicates(subset=['variant'], keep='first').copy().reset_index(drop=True))

    df = pd.concat(dfs).sort_values(by='pip', ascending=False).drop_duplicates(subset=['variant'], keep='first').copy().reset_index(drop=True)

    df['chromosome'] = "chr" + df['chromosome'].astype(str)
    df = df.rename(columns={'chromosome' : 'chrom', 'position' : 'pos'})
    
    print("len(df) = " + str(len(df)))

    #Save union of dataframes
    df.to_csv("txrev/GTEx_txrev_finemapped_merged.csv.gz", sep='\t', index=False)

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
