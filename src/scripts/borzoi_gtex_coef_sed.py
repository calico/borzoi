#!/usr/bin/env python
from optparse import OptionParser
import os
import pdb
import re
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

'''
borzoi_gtex_coef_sed.py

Evaluate concordance of variant effect prediction sign classifcation
and coefficient correlations (gene-specific).
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <gtex_dir>'
    parser = OptionParser(usage)
    
    parser.add_option(
        '-o',
        dest='out_dir',
        default='coef_out',
        help='Output directory for tissue metrics',
    )
    parser.add_option(
        '-g',
        dest='gtex_vcf_dir',
        default='susie_pip90',
        help='GTEx VCF directory',
    )
    parser.add_option(
        '--susie',
        dest='susie_dir',
        default='susie_pip90',
        help='SuSiE directory'
    )
    parser.add_option(
        '-m',
        dest='min_variants',
        type=int,
        default=32,
        help='Minimum number of variants for tissue to be included',
    )
    parser.add_option(
        '-p',
        dest='plot',
        default=False,
        action='store_true',
        help='Generate tissue prediction plots',
    )
    parser.add_option(
        '-s',
        dest='snp_stat',
        default='logSED',
        help='SNP statistic. [Default: %(default)s]',
    )
    parser.add_option(
        '-v',
        dest='verbose',
        default=False,
        action='store_true',
    )
    
    (options, args) = parser.parse_args()
    
    if len(args) != 1:
        parser.error('Must provide gtex output directory')
    else:
        gtex_dir = args[0]

    os.makedirs(options.out_dir, exist_ok=True)
    
    tissue_keywords = {
            'Adipose_Subcutaneous': 'adipose',
            'Adipose_Visceral_Omentum': 'adipose',
            'Adrenal_Gland': 'adrenal_gland',
            'Artery_Aorta': 'blood_vessel',
            'Artery_Coronary': 'blood_vessel',
            'Artery_Tibial': 'blood_vessel',
            'Brain_Amygdala' : 'brain',
            'Brain_Anterior_cingulate_cortex_BA24' : 'brain',
            'Brain_Caudate_basal_ganglia' : 'brain',
            'Brain_Cerebellar_Hemisphere' : 'brain',
            'Brain_Cerebellum': 'brain',
            'Brain_Cortex': 'brain',
            'Brain_Frontal_Cortex_BA9' : 'brain',
            'Brain_Hippocampus' : 'brain',
            'Brain_Hypothalamus' : 'brain',
            'Brain_Nucleus_accumbens_basal_ganglia' : 'brain',
            'Brain_Putamen_basal_ganglia' : 'brain',
            'Brain_Spinal_cord_cervical_c-1' : 'brain',
            'Brain_Substantia_nigra' : 'brain',
            'Breast_Mammary_Tissue': 'breast',
            'Cells_Cultured_fibroblasts' : 'skin',
            'Cells_EBV-transformed_lymphocytes' : 'blood',
            'Colon_Sigmoid': 'colon',
            'Colon_Transverse': 'colon',
            'Esophagus_Gastroesophageal_Junction' : 'esophagus',
            'Esophagus_Mucosa': 'esophagus',
            'Esophagus_Muscularis': 'esophagus',
            'Heart_Atrial_Appendage' : 'heart',
            'Heart_Left_Ventricle' : 'heart',
            'Kidney_Cortex' : 'kidney',
            'Liver': 'liver',
            'Lung': 'lung',
            'Minor_Salivary_Gland' : 'salivary_gland',
            'Muscle_Skeletal': 'muscle',
            'Nerve_Tibial': 'nerve',
            'Ovary': 'ovary',
            'Pancreas': 'pancreas',
            'Pituitary': 'pituitary',
            'Prostate': 'prostate',
            'Skin_Not_Sun_Exposed_Suprapubic': 'skin',
            'Skin_Sun_Exposed_Lower_leg' : 'skin',
            'Small_Intestine_Terminal_Ileum' : 'small_intestine',
            'Spleen': 'spleen',
            'Stomach': 'stomach',
            'Testis': 'testis',
            'Thyroid': 'thyroid',
            'Uterus' : 'uterus',
            'Vagina' : 'vagina',
            'Whole_Blood': 'blood',
    }
 
    metrics_tissue = []
    metrics_sauroc = []
    metrics_cauroc = []
    metrics_rs = []
    metrics_rp = []
    metrics_n = []
    for tissue, keyword in tissue_keywords.items():
        if options.verbose: print(tissue)

        # read causal variants
        eqtl_df = read_eqtl(tissue, options.gtex_vcf_dir, susie_dir=options.susie_dir)
        
        if eqtl_df is not None and eqtl_df.shape[0] > options.min_variants:
            # read model predictions
            gtex_scores_file = f'{gtex_dir}/{tissue}_pos/sed.h5'
            eqtl_df = add_scores(gtex_scores_file, keyword, eqtl_df,
                                                     options.snp_stat, verbose=options.verbose)

            eqtl_df = eqtl_df.loc[~eqtl_df['score'].isnull()].copy()
            
            # compute AUROCs
            sign_auroc = roc_auc_score(eqtl_df.coef > 0, eqtl_df.score)

            # compute SpearmanR
            coef_r = spearmanr(eqtl_df.coef, eqtl_df.score)[0]

            # compute PearsonR
            coef_rp = pearsonr(eqtl_df.coef, eqtl_df.score)[0]
            
            coef_n = len(eqtl_df)

            # classification AUROC
            class_auroc = classify_auroc(gtex_scores_file, keyword, eqtl_df,
                                                                     options.snp_stat)

            if options.plot:
                eqtl_df.to_csv(f'{options.out_dir}/{tissue}.tsv',
                                             index=False, sep='\t')

                # scatterplot
                plt.figure(figsize=(6,6))
                sns.scatterplot(x=eqtl_df.coef, y=eqtl_df.score,
                                                alpha=0.5, s=20)
                plt.gca().set_xlabel('eQTL coefficient')
                plt.gca().set_ylabel('Variant effect prediction')
                plt.savefig(f'{options.out_dir}/{tissue}.png', dpi=300)

            # save
            metrics_tissue.append(tissue)
            metrics_sauroc.append(sign_auroc)
            metrics_cauroc.append(class_auroc)
            metrics_rs.append(coef_r)
            metrics_rp.append(coef_rp)
            metrics_n.append(coef_n)

            if options.verbose: print('')

    # save metrics
    metrics_df = pd.DataFrame({
            'tissue': metrics_tissue,
            'auroc_sign': metrics_sauroc,
            'spearmanr': metrics_rs,
            'pearsonr': metrics_rp,
            'n': metrics_n,
            'auroc_class': metrics_cauroc
    })
    metrics_df.to_csv(f'{options.out_dir}/metrics.tsv',
                                        sep='\t', index=False, float_format='%.4f')

    # summarize
    print('Sign AUROC:    %.4f' % np.mean(metrics_df.auroc_sign))
    print('SpearmanR:     %.4f' % np.mean(metrics_df.spearmanr))
    print('Class AUROC: %.4f' % np.mean(metrics_df.auroc_class))


def read_eqtl(tissue: str, gtex_vcf_dir: str, pip_t: float=0.9, susie_dir: str='tissues_susie'):
    """Reads eQTLs from SUSIE output.
    
    Args:
        tissue (str): Tissue name.
        gtex_vcf_dir (str): GTEx VCF directory.
        pip_t (float): PIP threshold.

    Returns:
        eqtl_df (pd.DataFrame): eQTL dataframe, or None if tissue skipped.
    """

    # read causal variants
    eqtl_file = f'{susie_dir}/{tissue}.tsv'
    df_eqtl = pd.read_csv(eqtl_file, sep='\t', index_col=0)

    # pip filter
    pip_match = re.search(r"_pip(\d+).?$", gtex_vcf_dir).group(1)
    pip_t = float(pip_match) / 100
    assert(pip_t > 0 and pip_t <= 1)
    df_causal = df_eqtl[df_eqtl.pip > pip_t]
    
    # make table
    tissue_vcf_file = f'{gtex_vcf_dir}/{tissue}_pos.vcf'
    if not os.path.isfile(tissue_vcf_file):
        eqtl_df = None
    else:
        # create dataframe
        eqtl_df = pd.DataFrame({
            'variant': df_causal.variant,
            'gene': [trim_dot(gene_id) for gene_id in df_causal.gene],
            'coef': df_causal.beta_posterior,
            'allele1': df_causal.allele1
        })
    return eqtl_df


def add_scores(gtex_scores_file: str,
                             keyword: str,
                             eqtl_df: pd.DataFrame,
                             score_key: str='SED',
                             verbose: bool=False):
    """Read eQTL RNA predictions for the given tissue.
    
    Args:
        gtex_scores_file (str): Variant scores HDF5.
        tissue_keyword (str): tissue keyword, for matching GTEx targets
        eqtl_df (pd.DataFrame): eQTL dataframe
        score_key (str): score key in HDF5 file
        verbose (bool): Print matching targets.

    Returns:
        eqtl_df (pd.DataFrame): eQTL dataframe, with added scores
    """
    with h5py.File(gtex_scores_file, 'r') as gtex_scores_h5:
        # read data
        snp_i = gtex_scores_h5['si'][:]
        snps = np.array([snp.decode('UTF-8') for snp in gtex_scores_h5['snp']])
        ref_allele = np.array([ref.decode('UTF-8') for ref in gtex_scores_h5['ref_allele']])
        genes = np.array([snp.decode('UTF-8') for snp in gtex_scores_h5['gene']])
        target_ids = np.array([ref.decode('UTF-8') for ref in gtex_scores_h5['target_ids']])
        target_labels = np.array([ref.decode('UTF-8') for ref in gtex_scores_h5['target_labels']])
        
        # determine matching GTEx targets
        match_tis = []
        for ti in range(len(target_ids)):
            if target_ids[ti].find('GTEX') != -1 and target_labels[ti].find(keyword) != -1:
                if not keyword == 'blood' or target_labels[ti].find('vessel') == -1:
                    if verbose:
                        print(ti, target_ids[ti], target_labels[ti])
                    match_tis.append(ti)
        match_tis = np.array(match_tis)
        
        # read scores and take mean across targets
        score_table = gtex_scores_h5[score_key][...,match_tis].mean(axis=-1, dtype='float32')
        score_table = np.arcsinh(score_table)

    # hash scores to (snp,gene)
    snpgene_scores = {}
    for sgi in range(score_table.shape[0]):
        snp = snps[snp_i[sgi]]
        gene = trim_dot(genes[sgi])
        snpgene_scores[(snp,gene)] = score_table[sgi]

    # add scores to eQTL table
    #    flipping when allele1 doesn't match
    snp_ref = dict(zip(snps, ref_allele))
    eqtl_df_scores = []
    for sgi, eqtl in eqtl_df.iterrows():
        sgs = snpgene_scores.get((eqtl.variant,eqtl.gene), 0)
        if not np.isnan(sgs) and sgs != 0 and snp_ref[eqtl.variant] != eqtl.allele1:
            sgs *= -1
        eqtl_df_scores.append(sgs)
    eqtl_df['score'] = eqtl_df_scores

    return eqtl_df


def classify_auroc(gtex_scores_file: str,
                                     keyword: str,
                                     eqtl_df: pd.DataFrame,
                                     score_key: str='SED',
                                     agg_mode: str='max'):                             
    """Read eQTL RNA predictions for negatives from the given tissue.
    
    Args:
        gtex_scores_file (str): Variant scores HDF5.
        tissue_keyword (str): tissue keyword, for matching GTEx targets
        eqtl_df (pd.DataFrame): eQTL dataframe
        score_key (str): score key in HDF5 file
        verbose (bool): Print matching targets.

    Returns:
        class_auroc (float): Classification AUROC.
    """

    # read positive scores
    with h5py.File(gtex_scores_file, 'r') as gtex_scores_h5:
        # read data
        snp_i = gtex_scores_h5['si'][:]
        snps = np.array([snp.decode('UTF-8') for snp in gtex_scores_h5['snp']])
        genes = np.array([snp.decode('UTF-8') for snp in gtex_scores_h5['gene']])
        target_ids = np.array([ref.decode('UTF-8') for ref in gtex_scores_h5['target_ids']])
        target_labels = np.array([ref.decode('UTF-8') for ref in gtex_scores_h5['target_labels']])
        
        # determine matching GTEx targets
        match_tis = []
        for ti in range(len(target_ids)):
            if target_ids[ti].find('GTEX') != -1 and target_labels[ti].find(keyword) != -1:
                if not keyword == 'blood' or target_labels[ti].find('vessel') == -1:
                    match_tis.append(ti)
        match_tis = np.array(match_tis)
        
        # read scores and take mean across targets
        score_table = gtex_scores_h5[score_key][...,match_tis].mean(axis=-1, dtype='float32')
        score_table = np.arcsinh(score_table)
    
    # aggregate across genes (sum abs or max abs); positives
    psnp_scores = {}
    for sgi in range(score_table.shape[0]):
        snp = snps[snp_i[sgi]]
        if agg_mode == 'sum' :
            psnp_scores[snp] = psnp_scores.get(snp,0) + np.abs(score_table[sgi])
        elif agg_mode == 'max' :
            psnp_scores[snp] = max(psnp_scores.get(snp,0), np.abs(score_table[sgi]))
    
    # read negative scores
    gtex_nscores_file = gtex_scores_file.replace('_pos','_neg')
    with h5py.File(gtex_nscores_file, 'r') as gtex_scores_h5:
        # read data
        snp_i = gtex_scores_h5['si'][:]
        snps = np.array([snp.decode('UTF-8') for snp in gtex_scores_h5['snp']])
        genes = np.array([snp.decode('UTF-8') for snp in gtex_scores_h5['gene']])
        target_ids = np.array([ref.decode('UTF-8') for ref in gtex_scores_h5['target_ids']])
        target_labels = np.array([ref.decode('UTF-8') for ref in gtex_scores_h5['target_labels']])
        
        # determine matching GTEx targets
        match_tis = []
        for ti in range(len(target_ids)):
            if target_ids[ti].find('GTEX') != -1 and target_labels[ti].find(keyword) != -1:
                if not keyword == 'blood' or target_labels[ti].find('vessel') == -1:
                    match_tis.append(ti)
        match_tis = np.array(match_tis)
        
        # read scores and take mean across targets
        score_table = gtex_scores_h5[score_key][...,match_tis].mean(axis=-1, dtype='float32')
        score_table = np.arcsinh(score_table)

    # aggregate across genes (sum abs or max abs); negatives
    nsnp_scores = {}
    for sgi in range(score_table.shape[0]):
        snp = snps[snp_i[sgi]]
        if agg_mode == 'sum' :
            nsnp_scores[snp] = nsnp_scores.get(snp,0) + np.abs(score_table[sgi])
        elif agg_mode == 'max' :
            nsnp_scores[snp] = max(nsnp_scores.get(snp,0), np.abs(score_table[sgi]))

    # compute AUROC
    Xp = list(psnp_scores.values())
    Xn = list(nsnp_scores.values())
    X = Xp + Xn
    y = [1]*len(Xp) + [0]*len(Xn)
    
    return roc_auc_score(y, X)


def trim_dot(gene_id):
    """Trim dot off GENCODE id's."""
    dot_i = gene_id.rfind('.')
    if dot_i != -1:
        gene_id = gene_id[:dot_i]
    return gene_id


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
