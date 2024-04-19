#!/usr/bin/env python
from optparse import OptionParser

import glob
import os

import pandas as pd

import util

'''
make_vcfs.py

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

    pip = 0.2
    match_gene = 0
    match_allele = 1
    
    ################################################
    # intronic polyA QTLs
    
    out_dir = 'ipaqtl_pip%d%s%s' % (pip*100, 'g' if match_gene == 1 else 'e', 'a' if match_allele else '')
    os.makedirs(out_dir, exist_ok=True)

    jobs = []
    for table_file in glob.glob('txrev/*.txt.gz'):
        out_prefix = table_file.replace('txrev/', '%s/' % out_dir)
        out_prefix = out_prefix.replace('.purity_filtered.txt.gz', '')
        cmd = './ipaqtl_vcfs.py --neg_pip 0.01 --pos_pip %f --match_gene %d --match_allele %d -o %s' % (pip, match_gene, match_allele, out_prefix)
        jobs.append(cmd)
    util.exec_par(jobs, 6, verbose=True)

    # merge study/tissue variants
    mpos_vcf_file = '%s/pos_merge.vcf' % out_dir
    mneg_vcf_file = '%s/neg_merge.vcf' % out_dir
    merge_variants(mpos_vcf_file, '%s/*_pos.vcf' % out_dir)
    merge_variants(mneg_vcf_file, '%s/*_neg.vcf' % out_dir)
    
    
    ################################################
    # polyA QTLs

    out_dir = 'paqtl_pip%d%s%s' % (pip*100, 'g' if match_gene == 1 else 'e', 'a' if match_allele else '')
    os.makedirs(out_dir, exist_ok=True)

    jobs = []
    for table_file in glob.glob('txrev/*.txt.gz'):
        out_prefix = table_file.replace('txrev/', '%s/' % out_dir)
        out_prefix = out_prefix.replace('.purity_filtered.txt.gz', '')
        cmd = './paqtl_vcfs.py --neg_pip 0.01 --pos_pip %f --match_gene %d --match_allele %d -o %s' % (pip, match_gene, match_allele, out_prefix)
        jobs.append(cmd)
    util.exec_par(jobs, 6, verbose=True)

    # merge study/tissue variants
    mpos_vcf_file = '%s/pos_merge.vcf' % out_dir
    mneg_vcf_file = '%s/neg_merge.vcf' % out_dir
    merge_variants(mpos_vcf_file, '%s/*_pos.vcf' % out_dir)
    merge_variants(mneg_vcf_file, '%s/*_neg.vcf' % out_dir)
    
    ################################################
    # splicing QTLs
    
    out_dir = 'sqtl_pip%d%s%s' % (pip*100, 'g' if match_gene == 1 else 'e', 'a' if match_allele else '')
    os.makedirs(out_dir, exist_ok=True)

    jobs = []
    for table_file in glob.glob('txrev/*.txt.gz'):
        out_prefix = table_file.replace('txrev/', '%s/' % out_dir)
        out_prefix = out_prefix.replace('.purity_filtered.txt.gz', '')
        cmd = './sqtl_vcfs.py --neg_pip 0.01 --pos_pip %f --match_gene %d --match_allele %d -o %s' % (pip, match_gene, match_allele, out_prefix)
        jobs.append(cmd)
    util.exec_par(jobs, 6, verbose=True)

    # merge study/tissue variants
    mpos_vcf_file = '%s/pos_merge.vcf' % out_dir
    mneg_vcf_file = '%s/neg_merge.vcf' % out_dir
    merge_variants(mpos_vcf_file, '%s/*_pos.vcf' % out_dir)
    merge_variants(mneg_vcf_file, '%s/*_neg.vcf' % out_dir)
    

def merge_variants(merge_vcf_file, vcf_glob):
    with open(merge_vcf_file, 'w') as merge_vcf_open:
        vcf0_file = list(glob.glob(vcf_glob))[0]
        for line in open(vcf0_file):
            if line[0] == '#':
                print(line, end='', file=merge_vcf_open)

        merged_variants = set()
        for vcf_file in glob.glob(vcf_glob):
            for line in open(vcf_file):
                if not line.startswith('#'):
                    variant = line.split()[2]
                    if variant not in merged_variants:
                        print(line, file=merge_vcf_open, end='')
                        merged_variants.add(variant)

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
