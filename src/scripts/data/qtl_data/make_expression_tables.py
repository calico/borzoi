#!/usr/bin/env python
from optparse import OptionParser

import os

import util

import numpy as np
import pandas as pd

import pyranges as pr

import matplotlib.pyplot as plt

'''
make_expression_tables.py

Contruct TPM bucket to sample genes from.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    #parser.add_option()
    (options,args) = parser.parse_args()

    #Define tissue column-to-file mapping
    tissue_dict = {
        'Adipose - Subcutaneous' : 'adipose_subcutaneous',
        'Adipose - Visceral (Omentum)' : 'adipose_visceral',
        'Adrenal Gland' : 'adrenal_gland',
        'Artery - Aorta' : 'artery_aorta',
        'Artery - Coronary' : 'artery_coronary',
        'Artery - Tibial' : 'artery_tibial',
        'Whole Blood' : 'blood',
        'Brain - Amygdala' : 'brain_amygdala',
        'Brain - Anterior cingulate cortex (BA24)' : 'brain_anterior_cingulate_cortex',
        'Brain - Caudate (basal ganglia)' : 'brain_caudate',
        'Brain - Cerebellar Hemisphere' : 'brain_cerebellar_hemisphere',
        'Brain - Cerebellum' : 'brain_cerebellum',
        'Brain - Cortex' : 'brain_cortex',
        'Brain - Frontal Cortex (BA9)' : 'brain_frontal_cortex',
        'Brain - Hippocampus' : 'brain_hippocampus',
        'Brain - Hypothalamus' : 'brain_hypothalamus',
        'Brain - Nucleus accumbens (basal ganglia)' : 'brain_nucleus_accumbens',
        'Brain - Putamen (basal ganglia)' : 'brain_putamen',
        'Brain - Spinal cord (cervical c-1)' : 'brain_spinal_cord',
        'Brain - Substantia nigra' : 'brain_substantia_nigra',
        'Breast - Mammary Tissue' : 'breast',
        'Colon - Sigmoid' : 'colon_sigmoid',
        'Colon - Transverse' : 'colon_transverse',
        'Esophagus - Gastroesophageal Junction' : 'esophagus_gej',
        'Esophagus - Mucosa' : 'esophagus_mucosa',
        'Esophagus - Muscularis' : 'esophagus_muscularis',
        'Cells - Cultured fibroblasts' : 'fibroblast',
        'Heart - Atrial Appendage' : 'heart_atrial_appendage',
        'Heart - Left Ventricle' : 'heart_left_ventricle',
        'Kidney - Cortex' : 'kidney_cortex',
        'Cells - EBV-transformed lymphocytes' : 'LCL',
        'Liver' : 'liver',
        'Lung' : 'lung',
        'Minor Salivary Gland' : 'minor_salivary_gland',
        'Muscle - Skeletal' : 'muscle',
        'Nerve - Tibial' : 'nerve_tibial',
        'Ovary' : 'ovary',
        'Pancreas' : 'pancreas',
        'Pituitary' : 'pituitary',
        'Prostate' : 'prostate',
        'Skin - Not Sun Exposed (Suprapubic)' : 'skin_not_sun_exposed',
        'Skin - Sun Exposed (Lower leg)' : 'skin_sun_exposed',
        'Small Intestine - Terminal Ileum' : 'small_intestine',
        'Spleen' : 'spleen',
        'Stomach' : 'stomach',
        'Testis' : 'testis',
        'Thyroid' : 'thyroid',
        'Uterus' : 'uterus',
        'Vagina' : 'vagina',
    }
    
    for tissue_name in tissue_dict :
        
        #Load TPM matrix
        tpm_df = pd.read_csv("GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz", sep='\t', compression='gzip', skiprows=2)

        save_name = tissue_dict[tissue_name]

        print("-- " + save_name + " --")

        #Clean dataframe
        tpm_df['gene_id'] = tpm_df['Name'].apply(lambda x: x.split(".")[0])

        tpm_df = tpm_df.drop_duplicates(subset=['gene_id'], keep='first').copy().reset_index(drop=True)

        tpm_df['tpm'] = tpm_df[tissue_name]
        tpm_df = tpm_df[['gene_id', 'tpm']]

        #Get non-zero TPM entries
        tpm_df_zero = tpm_df.loc[tpm_df['tpm'] == 0].copy().reset_index(drop=True)
        tpm_df_nonzero = tpm_df.loc[tpm_df['tpm'] > 0].copy().reset_index(drop=True)

        tpm_df_zero['tpm_log2'] = 0.
        tpm_df_nonzero['tpm_log2'] = np.log2(tpm_df_nonzero['tpm'])

        #Clip at extremes
        min_q = 0.0075
        max_q = 0.9925

        #Log2 fold change bin sizes
        bin_size = 0.4
        bin_offset = 0.15

        min_tpm_log2 = np.quantile(tpm_df_nonzero['tpm_log2'], q=min_q)
        max_tpm_log2 = np.quantile(tpm_df_nonzero['tpm_log2'], q=max_q)

        tpm_df_nonzero.loc[tpm_df_nonzero['tpm_log2'] < min_tpm_log2, 'tpm_log2'] = min_tpm_log2
        tpm_df_nonzero.loc[tpm_df_nonzero['tpm_log2'] > max_tpm_log2, 'tpm_log2'] = max_tpm_log2

        tpm_log2 = tpm_df_nonzero['tpm_log2'].values

        n_bins = int((max_tpm_log2 - min_tpm_log2) / bin_size)

        #Get sample bins
        sample_bins = np.linspace(min_tpm_log2, max_tpm_log2, n_bins+1)

        #Map values to bins
        bin_index = np.digitize(tpm_log2, sample_bins[1:], right=True)
        bin_index_l = np.digitize(tpm_log2 - bin_offset, sample_bins[1:], right=True)
        bin_index_r = np.digitize(tpm_log2 + bin_offset, sample_bins[1:], right=True)

        tpm_df_zero['bin_index_l'] = -1 * np.ones(len(tpm_df_zero), dtype='int32')
        tpm_df_zero['bin_index'] = -1 * np.ones(len(tpm_df_zero), dtype='int32')
        tpm_df_zero['bin_index_r'] = -1 * np.ones(len(tpm_df_zero), dtype='int32')

        tpm_df_nonzero['bin_index_l'] = bin_index_l
        tpm_df_nonzero['bin_index'] = bin_index
        tpm_df_nonzero['bin_index_r'] = bin_index_r

        tpm_df = pd.concat([tpm_df_zero, tpm_df_nonzero]).copy().reset_index(drop=True)

        tpm_df = tpm_df.sort_values(by='gene_id', ascending=True).copy().reset_index(drop=True)

        #Save dataframe
        tpm_df.to_csv('ge/GTEx_ge_' + save_name + "_tpms.csv", sep='\t', index=False)

        #Visualize TPM sample bins
        tpm_df_filtered = tpm_df.loc[tpm_df['tpm'] > 0.]

        f = plt.figure(figsize=(4, 3))

        plt.hist(tpm_df_filtered['bin_index'].values, bins=np.unique(tpm_df_filtered['bin_index'].values))

        plt.xlim(0, np.max(tpm_df_filtered['bin_index'].values))

        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

        plt.xlabel("Sample bin (FC < " + str(round(2**(bin_size+2*bin_offset), 2)) + ")", fontsize=8)
        plt.ylabel("# of genes", fontsize=8)

        plt.title("TPM sample bins (" + save_name + ")", fontsize=8)

        plt.tight_layout()

        plt.savefig('ge/GTEx_ge_' + save_name + "_tpms.png", transparent=False, dpi=300)

        plt.close()

        #Check and warn in case of low-support bins
        _, bin_support = np.unique(tpm_df_filtered['bin_index'].values, return_counts=True)

        if np.any(bin_support < 100) :
            print("[Warning] Less than 100 genes in some of the TPM sample bins (min = " + str(int(np.min(bin_support))) + ").")

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
