#!/usr/bin/env python
from optparse import OptionParser

import os

import pandas as pd

import util

'''
download_finemap.py

Download QTL Catalogue fine-mapping results.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    #parser.add_option()
    (options,args) = parser.parse_args()

    # read remote table
    samples_df = pd.read_csv('https://raw.githubusercontent.com/eQTL-Catalogue/eQTL-Catalogue-resources/master/tabix/tabix_ftp_paths.tsv', sep='\t')

    # filter GTEx (for now)
    samples_df = samples_df[samples_df.study == 'GTEx']


    ################################################
    # txrevise for splicing / polyA / TSS QTLs

    os.makedirs('txrev', exist_ok=True)
    txrev_df = samples_df[samples_df.quant_method == 'txrev']

    jobs = []
    for all_ftp_path in txrev_df.ftp_path:
        # ftp://ftp.ebi.ac.uk/pub/databases/spot/eQTL/sumstats/Alasoo_2018/txrev/Alasoo_2018_txrev_macrophage_IFNg+Salmonella.all.tsv.gz
        # ftp://ftp.ebi.ac.uk/pub/databases/spot/eQTL/credible_sets//Alasoo_2018_txrev_macrophage_IFNg+Salmonella.purity_filtered.txt.gz

        all_ftp_file = all_ftp_path.split('/')[-1]
        fine_ftp_file = all_ftp_file.replace('all.tsv', 'purity_filtered.txt')

        fine_ftp_path = 'ftp://ftp.ebi.ac.uk/pub/databases/spot/eQTL/credible_sets/'
        fine_ftp_path += fine_ftp_file

        local_path = 'txrev/%s' % fine_ftp_file
        if not os.path.isfile(local_path):
            cmd = 'curl -o %s %s' % (local_path, fine_ftp_path)
            jobs.append(cmd)

    util.exec_par(jobs, 4, verbose=True)
    # print('\n'.join(jobs))


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
