#!/usr/bin/env python
from optparse import OptionParser
import os
import sys
import pyfaidx

'''
idx_genome.py

Create .fai index file for input .fa.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <genome_fa>'
    parser = OptionParser(usage)
    (options, args) = parser.parse_args()
    
    if len(args) != 1:
        parser.error('Must provide input fasta file')
    else:
        genome_fa = args[0]

    pyfaidx.Faidx(genome_fa)

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
