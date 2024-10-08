#!/usr/bin/env python
from optparse import OptionParser

from collections import OrderedDict
import os
import pdb
import sys

import h5py
from intervaltree import IntervalTree
import numpy as np
from scipy.ndimage.filters import maximum_filter1d
from sklearn.mixture import GaussianMixture

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

'''
w5_qc.py

Create a QC report for a Wig5 file.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <w5_file>'
    parser = OptionParser(usage)
    parser.add_option('-b', dest='blacklist_bed',
            default='/home/drk/common/data/genomes/hg38/blacklist/blacklist_hg38_all.bed',
            help='Blacklist BED file for annotating max regions.')
    parser.add_option('-c', dest='chrs',
            help='Process only the given comma-separated chromosomes')
    parser.add_option('-g', dest='genes_bed',
            help='Genes BED file for annotating max regions.')
    parser.add_option('-m', dest='max_pool',
            default=512, type='int',
            help='Max pool window to report max values')
    parser.add_option('-n', dest='max_n',
            default=100, type='int',
            help='Number of maximum coverage positions [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='w5_qc')
    parser.add_option('-p', dest='pool',
            default=32, type='int',
            help='Average pool window to reduce dimensionality [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 1:
        parser.error('Must provide Wig5.')
    else:
        w5_file = args[0]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    w5_open = h5py.File(w5_file, 'r')

    ############################################################
    # choose chromosomes

    if options.chrs is None:
        options.chrs = list(w5_open.keys())
        options.chrs = []
        for chrm in w5_open.keys():
            if chrm in ['chrM','chrEBV']:
                continue
            if chrm.startswith('chrUn'):
                continue
            if chrm.find('random') != -1:
                continue
            options.chrs.append(chrm)
    else:
        chrs_str = options.chrs
        options.chrs = []
        for chrm in chrs_str.split(','):
            if chrm in w5_open:
                options.chrs.append(chrm)
            else:
                print('Chromosome %s not found in %s' % (chrm, w5_file), file=sys.stderr)

    ############################################################
    # read genome coverage

    nan_out = open('%s/nan.txt' % options.out_dir, 'w')

    chr_lens = OrderedDict()
    genome_cov = []

    for chrm in options.chrs:
        # read chromosome coverage
        chr_cov = np.array(w5_open[chrm], dtype='float16')

        # truncate to fit reshape here and below
        pool_max = max(options.pool, options.max_pool)
        chr_mod = len(chr_cov) % pool_max
        chr_cov = chr_cov[:-chr_mod]

        # handle nan
        chr_nan = np.mean(np.isnan(chr_cov), dtype='float64')
        print('%-5s\t%7.2e' % (chrm, chr_nan), file=nan_out)
        chr_cov = np.nan_to_num(chr_cov)

        # save chromosome
        chr_lens[chrm] = len(chr_cov)

        # take means across windows
        chr_cov_pool = np.mean(np.reshape(chr_cov, (-1, options.pool)), axis=1)

        # append to genome coverage
        genome_cov.append(chr_cov_pool)

    genome_cov = np.concatenate(genome_cov)

    nan_out.close()

    ############################################################
    # plot distributions

    zero_mask = (genome_cov == 0)
    zero_pct = np.mean(zero_mask)

    zero_out = open('%s/zero.txt' % options.out_dir, 'w')
    print(zero_pct, file=zero_out)
    zero_out.close()

    sample_size = min((~zero_mask).sum(), 200000)
    sample_cov = np.random.choice(genome_cov[~zero_mask], size=sample_size, replace=False)

    plt.figure()
    sns.distplot(sample_cov)
    plt.savefig('%s/dist.pdf' % options.out_dir)
    plt.close()

    plt.figure()
    sns.distplot(np.sqrt(sample_cov))
    plt.savefig('%s/dist_sqrt.pdf' % options.out_dir)
    plt.close()

    plt.figure()
    sns.distplot(np.log(sample_cov+1))
    plt.savefig('%s/dist_log.pdf' % options.out_dir)
    plt.close()

    ############################################################
    # histogram values
    #  (which help identify sparse, poorly normalized files)

    # find largest chromosome
    chr_list = list(chr_lens.keys())
    lens_list = list(chr_lens.values())
    max_len_i = np.argmax(lens_list)
    max_chr = chr_list[max_len_i]

    # read coverage
    max_chr_cov = np.nan_to_num(w5_open[max_chr])

    # count values
    unique_cov, counts_cov = np.unique(max_chr_cov, return_counts=True)

    # write
    hist_out = open('%s/hist.txt' % options.out_dir, 'w')
    for i in range(len(unique_cov)):
        print('%-4d\t%7.4f\t%9d' % (i, unique_cov[i], counts_cov[i]), file=hist_out)
    hist_out.close()


    ############################################################
    # counts at thresholds

    counts_out = open('%s/tcounts.txt' % options.out_dir, 'w')
    for t in [4, 8, 16, 32, 64, 128, 256, 512]:
        tcount = np.sum(genome_cov > t)
        tpct = np.mean(genome_cov > t)
        print('%-3d\t%8d\t%.2e' % (t, tcount, tpct), file=counts_out)
    counts_out.close()

    ############################################################
    # compute genome percentiles

    pcts = np.array([.001, .01, .05, .25, .50, .75, .95, .99, .999])
    cov_pcts = np.percentile(genome_cov, 100*pcts)

    pcts_out = open('%s/percentiles.txt' % options.out_dir, 'w')
    for i in range(len(pcts)):
        print('%5.3f\t%7.3f' % (pcts[i], cov_pcts[i]), file=pcts_out)
    pcts_out.close()

    ############################################################
    # compute genome and chromosome means

    means_out = open('%s/means.txt' % options.out_dir, 'w')

    genome_cov_mean = np.mean(genome_cov, dtype='float64')
    print('%-5s\t%9d\t%6f\t%5.3f' % ('whole', 1, genome_cov_mean, 1.0), file=means_out)

    for chrm in options.chrs:
        chr_cov = np.nan_to_num(w5_open[chrm])

        # compute chromosome coverage mean and ratio
        chr_cov_mean = np.mean(chr_cov, dtype='float64')
        chr_ratio = chr_cov_mean / genome_cov_mean
        print('%-23s\t%9d\t%6f\t%5.3f' % (chrm, len(chr_cov), chr_cov_mean, chr_ratio), file=means_out)

    means_out.close()

    ############################################################
    # compute genome and chromosome means

    # blacklist annotation
    blacklist_trees = bed_chr_trees(options.blacklist_bed)

    # genes annotation
    gene_trees = bed_chr_trees(options.genes_bed)

    # reshape
    pool_mod = options.max_pool % options.pool
    if pool_mod != 0:
        old_pool = options.max_pool
        options.max_pool -= pool_mod
        print('Modifying max pool %d to %d to be divisible with avg pool.' % (old_pool, options.max_pool), file=sys.stderr)
    add_pool = options.max_pool // options.pool
    genome_cov_maxp = np.max(np.reshape(genome_cov, (-1,add_pool)), axis=1)

    max_out = open('%s/max.txt' % options.out_dir, 'w')

    mi = 0
    while mi < options.max_n:
        max_i = np.argmax(genome_cov_maxp)
        genome_i = max_i*options.max_pool
        chrm, pos = genome_chr_pos(genome_i, chr_lens)

        annotations = []

        # annotate blacklist
        blacklist_chr_tree = blacklist_trees.get(chrm, IntervalTree())
        if blacklist_chr_tree[pos:pos+options.max_pool]:
            annotations.append('blacklist')

        # annotate genes
        gene_chr_tree = gene_trees.get(chrm, IntervalTree())
        if gene_chr_tree[pos:pos+options.max_pool]:
            annotations.append('gene')

        ann_str = ','.join(annotations)

        print('%-5s\t%9d\t%7f\t%s' % (chrm, pos, genome_cov_maxp[max_i], ann_str), file=max_out)

        # zero the coverage so we don't pick it again
        genome_cov_maxp[max_i] = 0

        # next max
        mi += 1

    max_out.close()

    w5_open.close()


def bed_chr_trees(bed_file):
    """Return a dict mapping chromosomes to IntervalTrees."""
    chr_trees = {}
    if bed_file is not None:
        for line in open(bed_file):
            a = line.split()
            chrm = a[0]
            start = int(a[1])
            end = int(a[2])

            if chrm not in chr_trees:
                chr_trees[chrm] = IntervalTree()

            chr_trees[chrm][start:end] = True

    return chr_trees


def genome_chr_pos(gi, chr_lens):
    """ Compute chromosome and position for a genome index.

        Args
         gi (int): Genomic index
         chr_lens (OrderedDict): Chromosome lengths

        Returns:
         chrm (str): Chromosome
         pos (int): Position
        """

    chrms_list = list(chr_lens.keys())
    lengths_list = list(chr_lens.values())

    # chromosome index
    ci = 0

    # helper counters
    gii = 0
    cii = 0

    # while gi is beyond this chromosome
    while ci < len(lengths_list) and gi - gii > lengths_list[ci]:
      # advance genome index
      gii += lengths_list[ci]

      # advance chromosome
      ci += 1

    # we shouldn't be beyond the chromosomes
    assert (ci < len(lengths_list))

    # set position
    pos = gi - gii

    return chrms_list[ci], pos

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
