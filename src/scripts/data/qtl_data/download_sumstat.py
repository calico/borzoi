#!/usr/bin/env python
from optparse import OptionParser

import os

import pandas as pd

import util

'''
download_sumstat.py

Download QTL Catalogue sumstats.
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
    # ge for sumstat (we want SNPs and possibly also base expression)

    os.makedirs('ge', exist_ok=True)
    txrev_df = samples_df[samples_df.quant_method == 'ge']

    jobs = []
    for all_ftp_path in txrev_df.ftp_path:
        # ftp://ftp.ebi.ac.uk/pub/databases/spot/eQTL/sumstats/Alasoo_2018/txrev/Alasoo_2018_txrev_macrophage_IFNg+Salmonella.all.tsv.gz
        
        local_path = 'ge/%s' % all_ftp_path.split("/")[-1]
        
        if not os.path.isfile(local_path):
            cmd = 'curl -o %s %s' % (local_path, all_ftp_path)
            jobs.append(cmd)

    util.exec_par(jobs, 4, verbose=True)
    # print('\n'.join(jobs))


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
