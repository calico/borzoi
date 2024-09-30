#!/usr/bin/env python
from optparse import OptionParser

import gzip
import pdb

'''
pygene

Classes and methods to manage genes in GTF format.
'''

################################################################################
# Classes
################################################################################
class GenomicInterval:
  def __init__(self, start, end, chrom=None, strand=None):
    self.start = start
    self.end = end
    self.chrom = chrom
    self.strand = strand

  def __eq__(self, other):
    return self.start == other.start

  def __lt__(self, other):
    return self.start < other.start

  def __cmp__(self, x):
    if self.start < x.start:
      return -1
    elif self.start > x.start:
      return 1
    else:
      return 0

  def __str__(self):
    if self.chrom is None:
      label = '[%d-%d]' % (self.start, self.end)
    else:
      label =  '%s:%d-%d' % (self.chrom, self.start, self.end)
    return label


class Transcript:
  def __init__(self, chrom, strand, kv):
    self.chrom = chrom
    self.strand = strand
    self.kv = kv
    self.exons = []
    self.cds = []
    self.utrs3 = []
    self.utrs5 = []
    self.sorted = False
    self.utrs_defined = False

  def add_cds(self, start, end):
    self.cds.append(GenomicInterval(start,end))

  def add_exon(self, start, end):
    self.exons.append(GenomicInterval(start,end))

  def define_utrs(self):
    self.utrs_defined = True

    if len(self.cds) == 0:
      self.utrs3 = self.exons

    else:
      assert(self.sorted)

      # reset UTR lists
      self.utrs5 = []
      self.utrs3 = []

      # match up exons and CDS
      ci = 0
      for ei in range(len(self.exons)):
        # left initial
        if self.exons[ei].end < self.cds[ci].start:
          utr = GenomicInterval(self.exons[ei].start, self.exons[ei].end)
          if self.strand == '+':
            self.utrs5.append(utr)
          else:
            self.utrs3.append(utr)

        # right initial
        elif self.cds[ci].end < self.exons[ei].start:
          utr = GenomicInterval(self.exons[ei].start, self.exons[ei].end)
          if self.strand == '+':
            self.utrs3.append(utr)
          else:
            self.utrs5.append(utr)

        # overlap
        else:
          # left overlap
          if self.exons[ei].start < self.cds[ci].start:
            utr = GenomicInterval(self.exons[ei].start, self.cds[ci].start-1)
            if self.strand == '+':
              self.utrs5.append(utr)
            else:
              self.utrs3.append(utr)

          # right overlap
          if self.cds[ci].end < self.exons[ei].end:
            utr = GenomicInterval(self.cds[ci].end+1, self.exons[ei].end)
            if self.strand == '+':
              self.utrs3.append(utr)
            else:
              self.utrs5.append(utr)

          # increment up to last
          ci = min(ci+1, len(self.cds)-1)

  def fasta_cds(self, fasta_open, stranded=False):
    assert(self.sorted)
    gene_seq = ''
    for exon in self.cds:
      exon_seq = fasta_open.fetch(self.chrom, exon.start-1, exon.end)
      gene_seq += exon_seq
    if stranded and self.strand == '-':
      gene_seq = rc(gene_seq)
    return gene_seq

  def fasta_exons(self, fasta_open, stranded=False):
    assert(self.sorted)
    gene_seq = ''
    for exon in self.exons:
      exon_seq = fasta_open.fetch(self.chrom, exon.start-1, exon.end)
      gene_seq += exon_seq
    if stranded and self.strand == '-':
      gene_seq = rc(gene_seq)
    return gene_seq

  def sort_exons(self):
    self.sorted = True
    if len(self.exons) > 1:
      self.exons.sort()
    if len(self.cds) > 1:
      self.cds.sort()

  def span(self):
    exon_starts = [exon.start for exon in self.exons]
    exon_ends = [exon.end for exon in self.exons]
    return min(exon_starts), max(exon_ends)

  def tss(self):
    if self.strand == '-':
      return self.exons[-1].end
    else:
      return self.exons[0].start

  def write_gtf(self, gtf_out, write_cds=False, write_utrs=False):
    for ex in self.exons:
      cols = [self.chrom, 'pygene', 'exon', str(ex.start), str(ex.end)]
      cols += ['.', self.strand, '.', kv_gtf(self.kv)]
      print('\t'.join(cols), file=gtf_out)
    if write_cds:
      for cds in self.cds:
        cols = [self.chrom, 'pygene', 'CDS', str(cds.start), str(cds.end)]
        cols += ['.', self.strand, '.', kv_gtf(self.kv)]
        print('\t'.join(cols), file=gtf_out)
    if write_utrs:
      assert(self.utrs_defined)
      for utr in self.utrs5:
        cols = [self.chrom, 'pygene', '5\'UTR', str(utr.start), str(utr.end)]
        cols += ['.', self.strand, '.', kv_gtf(self.kv)]
        print('\t'.join(cols), file=gtf_out)
      for utr in self.utrs3:
        cols = [self.chrom, 'pygene', '3\'UTR', str(utr.start), str(utr.end)]
        cols += ['.', self.strand, '.', kv_gtf(self.kv)]
        print('\t'.join(cols), file=gtf_out)

  def __str__(self):
    return '%s %s %s %s' % (self.chrom, self.strand, kv_gtf(self.kv), ','.join([ex.__str__() for ex in self.exons]))


class Gene:
  def __init__(self):
    self.transcripts = {}
    self.chrom = None
    self.strand = None
    self.start = None
    self.end = None

  def add_transcript(self, tx_id, tx):
    self.transcripts[tx_id] = tx
    self.chrom = tx.chrom
    self.strand = tx.strand
    self.kv = tx.kv

  def span(self):
    tx_spans = [tx.span() for tx in self.transcripts.values()]
    tx_starts, tx_ends = zip(*tx_spans)
    self.start = min(tx_starts)
    self.end = max(tx_ends)
    return self.start, self.end


class GTF:
  def __init__(self, gtf_file, trim_dot=False):
    self.gtf_file = gtf_file
    self.genes = {}
    self.transcripts = {}
    self.utrs_defined = False
    self.trim_dot = trim_dot

    self.read_gtf()

  def define_utrs(self):
    self.utrs_defined = True
    for tx in self.transcripts.values():
      tx.define_utrs()

  def read_gtf(self):
    if self.gtf_file[-3:] == '.gz':
      gtf_in = gzip.open(self.gtf_file, 'rt')
    else:   
      gtf_in = open(self.gtf_file)

    # ignore header
    line = gtf_in.readline()
    while line[0] == '#':
        line = gtf_in.readline()

    while line:
      a = line.split('\t')
      if a[2] in ['exon','CDS']:
        chrom = a[0]
        interval_type = a[2]
        start = int(a[3])
        end = int(a[4])
        strand = a[6]
        kv = gtf_kv(a[8])

        # add/get transcript
        tx_id = kv['transcript_id']
        if self.trim_dot:
          tx_id = trim_dot(tx_id)
        if not tx_id in self.transcripts:
            self.transcripts[tx_id] = Transcript(chrom, strand, kv)
        tx = self.transcripts[tx_id]

        # add/get gene
        gene_id = kv['gene_id']
        if self.trim_dot:
          gene_id = trim_dot(gene_id)
        if not gene_id in self.genes:
          self.genes[gene_id] = Gene()
        self.genes[gene_id].add_transcript(tx_id, tx)

        # add exons
        if interval_type == 'exon':
          tx.add_exon(start, end)
        elif interval_type == 'CDS':
          tx.add_cds(start, end)

      line = gtf_in.readline()

    gtf_in.close()

    # sort transcript exons
    for tx in self.transcripts.values():
      tx.sort_exons()

  def write_gtf(self, out_gtf_file, write_cds=False, write_utrs=False):
    if write_utrs and not self.utrs_defined:
      self.define_utrs()

    gtf_out = open(out_gtf_file, 'w')
    for tx in self.transcripts.values():
      tx.write_gtf(gtf_out, write_cds, write_utrs)
    gtf_out.close()


################################################################################
# Methods
################################################################################
def gtf_kv(s):
  """Convert the last gtf section of key/value pairs into a dict."""
  d = {}

  a = s.split(';')
  for key_val in a:
    if key_val.strip():
      eq_i = key_val.find('=')
      if eq_i != -1 and key_val[eq_i-1] != '"':
        kvs = key_val.split('=')
      else:
        kvs = key_val.split()

      key = kvs[0]
      if kvs[1][0] == '"' and kvs[-1][-1] == '"':
        val = (' '.join(kvs[1:]))[1:-1].strip()
      else:
        val = (' '.join(kvs[1:])).strip()

      d[key] = val

  return d

def kv_gtf(d):
  """Convert a kv hash to str gtf representation."""
  s = ''

  if 'gene_id' in d.keys():
    s += '%s "%s"; ' % ('gene_id',d['gene_id'])

  if 'transcript_id' in d.keys():
    s += '%s "%s"; ' % ('transcript_id',d['transcript_id'])

  for key in sorted(d.keys()):
    if key not in ['gene_id','transcript_id']:
      s += '%s "%s"; ' % (key,d[key])

  return s

def trim_dot(gene_id):
  """Trim the final dot suffix off a gene_id."""
  dot_i = gene_id.rfind('.')
  if dot_i != -1:
    gene_id = gene_id[:dot_i]
  return gene_id