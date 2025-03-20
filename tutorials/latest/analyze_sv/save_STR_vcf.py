
import re
import os
import argparse
import numpy as np
import pandas as pd
from Bio import SeqIO


def strip_tissue(tissues):
    tissue_list = []
    for tissue in tissues:
        tissue_new = tissue.split("_")[0]
        tissue_list.append(tissue_new)
    return tissue_list

def strip_score(tissues):
    score_list = []
    for tissue in tissues:
        score = tissue.split("_")[1]
        score_list.append(float(score))
    if len(score_list) == 1:
        return True
    else:
        # check if signs of all scores are the same
        if all(x > 0 for x in score_list) or all(x < 0 for x in score_list):
            return True
        else:
            return False

# find tissue with the highest score
def max_tissue(tissues):
    score_list = []
    for tissue in tissues:
        score = tissue.split("_")[1]
        score_list.append(float(score))
    max_index = score_list.index(max(score_list))
    tissue_clean = tissues[max_index].split("_")[0]
    return tissue_clean

# find motif occurence numbers with regex
def find_motif(seq_dict, coords, motif):
    seq_to_search = seq_dict[coords[0]][coords[1]:coords[2]].upper()
    motif_dict = []
    if len(motif)>1:
        matches = re.finditer(motif, seq_to_search)
        for match in matches:
            start = match.start()
            end = match.end()
            motif_dict.append((coords[1]+start, coords[1]+end))
    else:
        if seq_to_search==motif*len(seq_to_search):
            for i in range(len(seq_to_search)):
                motif_dict.append((coords[1]+i, coords[1]+i+1))

    return motif_dict


def save_to_vcf(df, seq_dict, args):

    reduce_motifs = args.reduce
    extend_motifs = args.extend

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    names_vcf = []
    arr_repeats = []

    for index, row in df.iterrows():
        
        chrom = row['chrom']
        start = row['str.start']-1
        end = row['str.end']
        num_motifs = row['num_motifs']
        first_start = row['motif_coords_0'][0][0]
        first_end = row['motif_coords_0'][0][1]
        last_end = row['motif_coords_0'][-1][1]
        motif_coords = row['motif_coords_0']
        partial_start = row['start_partial']
        partial_end = row['end_partial']

        ref_allele_full = seq_dict[chrom][start:end].upper()
        motif = row['str.motif.forward'].upper()

        range_repeats = []
        if num_motifs-reduce_motifs>1:
            range_repeats.extend(np.arange(num_motifs-reduce_motifs, num_motifs))
        else:
            range_repeats.extend(np.arange(1, num_motifs))
        range_repeats.extend(np.arange(num_motifs+1, num_motifs+extend_motifs))
        
        for repeat in range_repeats:
            # if number of repeats is less than num_motifs, it's a deletion
            if repeat<num_motifs:
                # deletion
                if not partial_start:
                    ref_allele = seq_dict[chrom][start-1:start].upper()
                    alt_allele = ref_allele
                    pos_vcf = start
                else:
                    ref_allele = seq_dict[chrom][first_start-1:first_start].upper()
                    alt_allele = ref_allele
                    pos_vcf = first_start
                # add number of repeats to be deleted
                ref_allele += motif*(num_motifs-repeat)
            else:
                # insertion
                if not partial_start:
                    ref_allele = seq_dict[chrom][start-1:start].upper()
                    alt_allele = ref_allele
                    pos_vcf = start
                else:
                    ref_allele = seq_dict[chrom][first_start-1:first_start].upper()
                    alt_allele = ref_allele
                    pos_vcf = first_start
                # add number of repeats to be inserted
                alt_allele += motif*(repeat-num_motifs)

            vcf_df = pd.DataFrame({'chr': [chrom], 'pos': [pos_vcf], 'snp': [f"{chrom}_{pos_vcf}_{ref_allele}_{alt_allele}_b19"],
                                'ref': [ref_allele], 'alt': [alt_allele], 'x1': ['.'], 'x2': ['.']})
            vcf_df.to_csv(f'{args.output_dir}/{chrom}_{pos_vcf}_{repeat}.vcf', header=None, index=None, sep='\t')


def main():

    parser = argparse.ArgumentParser(description="Save STRs as VCF files")
    parser.add_argument("--input", required=True, help=".csv file containing STRs")
    parser.add_argument("--fasta", required=True, help="Reference genome FASTA file")
    parser.add_argument("--reduce", type=int, default=4, help="Reduce the number of repeats by this number")
    parser.add_argument("--extend", type=int, default=8, help="Extend the number of repeats by this number")
    parser.add_argument("--output_dir", default="output", help="Output directory")
    
    args = parser.parse_args()

    df = pd.read_csv(args.input, header=0)
    df['tissue_list'] = [strip_tissue(x.split(';')) for x in df['tissue_info']]
    df['score_concord'] = [strip_score(x.split(';')) for x in df['tissue_info']]
    df['max_tissue'] = [max_tissue(x.split(';')) for x in df['tissue_info']]
    df['num_tissues'] = [len(x) for x in df['tissue_list']]

    # filter and retain only rows with 'score'>0.25 and betas concordant between tissues
    df = df[df['score']>0.25]
    df = df[df['score_concord']==True]

    # dictionary to store hg19 sequences
    seq_dict = {}

    with open(args.fasta, mode="r") as handle:
        # process each record in .fa file if there's more than one
        for record in SeqIO.parse(handle, "fasta"):
            identifier = record.id
            sequence = record.seq
            seq_dict[identifier] = str(sequence)

    # parse sequences chrom:start-end from hg19 
    num_motifs, motif_coords, start_partial, end_partial = [], [], [], []

    for index, row in df.iterrows():
        chrom = row['chrom']
        start = row['str.start']-1
        end = row['str.end']
        coords = (chrom, start, end)
        motif = row['str.motif.forward'].upper()
        motif_dict = find_motif(seq_dict, coords, motif)
        if len(motif_dict)>0:
            if motif_dict[0][0]==start:
                start_partial.append(False)
            else:
                start_partial.append(True)
            if motif_dict[-1][1]==end:
                end_partial.append(False)
            else:
                end_partial.append(True)
        else:
            start_partial.append(False)
            end_partial.append(False)
        num_motifs.append(len(motif_dict))
        motif_coords.append(motif_dict)

    df['num_motifs'], df['motif_coords_0'], df['start_partial'], df['end_partial'] = num_motifs, motif_coords, start_partial, end_partial

    # filter and retain only rows with >0 motifs
    df = df[df['num_motifs']>0]
    df['tissues'] = [','.join(x) for x in df['tissue_list']]

    # save to vcf
    save_to_vcf(df, seq_dict, args)


if __name__ == "__main__":
    main()