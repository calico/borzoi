#!/usr/bin/env python
# Copyright 2023 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#         https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from optparse import OptionParser
import glob
import os
import pdb
import pickle
import shutil
import subprocess
import sys

import h5py
import numpy as np

import pandas as pd
import slurm

"""
borzoi_apa_ism_cov3_multi.py

Run UTR-wide APA ISM analysis, using multiple processes.
"""

################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] <params> <model> <gene_csv>"
    parser = OptionParser(usage)

    # ism
    parser.add_option(
        "--fa",
        dest="genome_fasta",
        default="%s/assembly/ucsc/hg38.fa" % os.environ.get("BORZOI_HG38", "hg38"),
        help="Genome FASTA for sequences [Default: %default]",
    )
    parser.add_option(
        "-o",
        dest="out_dir",
        default="satg_out",
        help="Output directory [Default: %default]",
    )
    parser.add_option(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Ensemble forward and reverse complement predictions [Default: %default]",
    )
    parser.add_option(
        "--separate_rc",
        dest="separate_rc",
        default=False,
        action="store_true",
        help="Store reverse complement scores separately (do not average with forward scores) [Default: %default]",
    )
    parser.add_option(
        "--head",
        dest="head_i",
        default=None,
        type="int",
        help="Parameters head [Default: %default]",
    )
    parser.add_option(
        "-f",
        dest="folds",
        default="0",
        type="str",
        help="Model folds to use in ensemble [Default: %default]",
    )
    parser.add_option(
        "-c",
        dest="crosses",
        default="0",
        type="str",
        help="Model crosses to use in ensemble [Default: %default]",
    )
    parser.add_option(
        "--shifts",
        dest="shifts",
        default="0",
        type="str",
        help="Ensemble prediction shifts [Default: %default]",
    )
    parser.add_option(
        "--paext",
        dest="pas_ext",
        default=100,
        type="int",
        help="Extension in bp past UTR span annotation [Default: %default]"
    )
    parser.add_option(
        "--upstream_paext",
        dest="pas_ext_up",
        default=None,
        type="int",
        help="Extension in bp past UTR span annotation, upstream only [Default: %default]",
    )
    parser.add_option(
        "-t",
        dest="targets_file",
        default=None,
        type="str",
        help="File specifying target indexes and labels in table format",
    )
    parser.add_option(
        "--ism_size",
        dest="ism_size",
        default=192,
        type="int",
        help="Length of sequence window to run ISM across (centered at pA site).",
    )
    parser.add_option(
        "--splice_ism_size",
        dest="splice_ism_size",
        default=64,
        type="int",
        help="Length of sequence window to run ISM across (centered at splice site).",
    )
    parser.add_option(
        "--do_shuffle",
        dest="do_shuffle",
        default=False,
        action="store_true",
        help="Run ISM Shuffle (otherwise ISM) [Default: %default]",
    )    
    parser.add_option(
        "--window_size",
        dest="window_size",
        default=1,
        type="int",
        help="ISM shuffle window size [Default: %default]",
    )
    parser.add_option(
        "--n_samples",
        dest="n_samples",
        default=8,
        type="int",
        help="ISM shuffle samples per position [Default: %default]",
    )
    parser.add_option(
        "--mononuc_shuffle",
        dest="mononuc_shuffle",
        default=False,
        action="store_true",
        help="Mono-nucleotide shuffle [Default: %default]",
    )
    parser.add_option(
        "--dinuc_shuffle",
        dest="dinuc_shuffle",
        default=False,
        action="store_true",
        help="Di-nucleotide shuffle [Default: %default]",
    )
    parser.add_option(
        "--pseudo",
        dest="pseudo",
        default=None,
        type="float",
        help="Constant pseudo count [Default: %default]",
    )
    parser.add_option(
        "--full_utr",
        dest="full_utr",
        default=False,
        action="store_true",
        help="Run ISM over full UTR [Default: %default]",
    )
    parser.add_option(
        "--apa_file",
        dest="apa_file",
        default=None,
        type="str",
        help="File specifying APA sites",
    )
    parser.add_option(
        "--splice_file",
        dest="splice_file",
        default=None,
        type="str",
        help="File specifying Splice sites",
    )
    parser.add_option(
        "-u",
        dest="untransform_old",
        default=False,
        action="store_true",
    )

    # multi
    parser.add_option(
        "-e",
        dest="conda_env",
        default="tf28",
        help="Anaconda environment [Default: %default]",
    )
    parser.add_option(
        "--name",
        dest="name",
        default="apa-ism",
        help="SLURM name prefix [Default: %default]",
    )
    parser.add_option(
        "--max_proc",
        dest="processes",
        default=None,
        type="int",
        help="Number parallel processes [Default: %default]",
    )
    parser.add_option(
        "-r",
        dest="genes_per_chunk",
        default=8,
        type="int",
        help="Genes per thread/chunk [Default: %default]",
    )
    parser.add_option(
        "-q",
        dest="queue",
        default="geforce",
        help="SLURM queue on which to run the jobs [Default: %default]",
    )
    parser.add_option(
        "--restart",
        dest="restart",
        default=False,
        action="store_true",
        help="Restart a partially completed job [Default: %default]"
    )
    (options, args) = parser.parse_args()

    if len(args) == 3 :
        params_file = args[0]
        model_folder = args[1]
        genes_csv_file = args[2]
    else:
        parser.error("Must provide parameter file, model folder and csv file")

    #######################################################
    # prep work

    # output directory
    if not options.restart:
        if os.path.isdir(options.out_dir):
            print("Please remove %s" % options.out_dir, file=sys.stderr)
            exit(1)
        os.mkdir(options.out_dir)
    
    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)
    
    fold_list = [int(fold) for fold in options.folds.split(",")]
    cross_list = [int(cross) for cross in options.crosses.split(",")]
    
    max_fold_ix = np.max(fold_list)
    max_cross_ix = np.max(cross_list)
    
    #################################################################
    # gene metadata
    
    # load gene dataframe
    gene_df = pd.read_csv(genes_csv_file, compression="gzip", sep="\t")
    n_genes = len(gene_df)
    
    #######################################################
    # launch parallel ism jobs
    
    # command base
    cmd_base = ('. %s; ' % os.environ['BORZOI_CONDA']) if 'BORZOI_CONDA' in os.environ else ''
    cmd_base += "conda activate %s;" % options.conda_env
    cmd_base += " echo $HOSTNAME;"
    
    shuffle_str = ""
    if options.do_shuffle :
        shuffle_str = "_shuffle"
    
    jobs = []

    chunk_i = 0
    chunk_start = 0
    chunk_end = min(chunk_start+options.genes_per_chunk, n_genes)

    while chunk_start < n_genes:
        
        cmd = cmd_base + " time borzoi_apa_ism_cov3.py"
        
        # parallelization parameters
        cmd += " -s %d" % chunk_start
        cmd += " -e %d" % chunk_end
        cmd += " -p %d" % chunk_i
        
        # ism parameters
        cmd += " --fa %s" % options.genome_fasta
        cmd += " -o %s" % options.out_dir
        if options.rc :
            cmd += " --rc"
        if options.head_i is not None :
            cmd += " --head %d" % options.head_i
        if options.separate_rc :
            cmd += " --separate_rc"
        cmd += " -f %s" % options.folds
        cmd += " -c %s" % options.crosses
        cmd += " --shifts %s" % options.shifts
        cmd += " --paext %d" % options.pas_ext
        if options.pas_ext_up is not None :
            cmd += " --upstream_paext %d" % options.pas_ext_up
        if options.targets_file is not None :
            cmd += " -t %s" % options.targets_file
        cmd += " --ism_size %d" % options.ism_size
        cmd += " --splice_ism_size %d" % options.splice_ism_size
        if options.do_shuffle :
            cmd += " --do_shuffle"
        cmd += " --window_size %d" % options.window_size
        cmd += " --n_samples %d" % options.n_samples
        if options.mononuc_shuffle :
            cmd += " --mononuc_shuffle"
        if options.dinuc_shuffle :
            cmd += " --dinuc_shuffle"
        if options.pseudo is not None :
            cmd += " --pseudo %d" % options.pseudo
        if options.full_utr :
            cmd += " --full_utr"
        if options.apa_file is not None :
            cmd += " --apa_file %s" % options.apa_file
        if options.splice_file is not None :
            cmd += " --splice_file %s" % options.splice_file
        if options.untransform_old :
            cmd += " -u"
        
        # script arguments
        cmd += " %s %s %s" % (params_file, model_folder, genes_csv_file)
        
        success_file = "%s/ism%s_part%d-success.txt" % (options.out_dir, shuffle_str, chunk_i)
        success_file = '%s/ism%s_f%dc%d_part%d-success.txt' % (options.out_dir, shuffle_str, max_fold_ix, max_cross_ix, chunk_i)
        
        if not os.path.isfile(success_file):
            j = slurm.Job(cmd,
                    name="%s_%d" % (options.name, chunk_i),
                    out_file="%s/%s_%d.out" % (options.out_dir, options.name, chunk_i),
                    err_file="%s/%s_%d.err" % (options.out_dir, options.name, chunk_i),
                    queue=options.queue, gpu=1, mem=45000, time="30-0:0:0")
            jobs.append(j)
        else:
            print('Skipping existing %s/ism%s_f%dc%d_part%d.h5' % (options.out_dir, shuffle_str, max_fold_ix, max_cross_ix, chunk_i), file=sys.stderr)

        # update
        chunk_i += 1
        chunk_start += options.genes_per_chunk
        chunk_end = min(chunk_start+options.genes_per_chunk, n_genes)

    slurm.multi_run(jobs, max_proc=options.processes, verbose=True, launch_sleep=10, update_sleep=60)

    #######################################################
    # collect output (per fold and cross)

    for fold_ix in fold_list :
        for cross_ix in cross_list :
            collect_h5("%s/ism%s_f%dc%d.h5" % (options.out_dir, shuffle_str, fold_ix, cross_ix), shuffle_str, fold_ix, cross_ix, options.out_dir, chunk_i, n_genes)


def collect_h5(file_name, shuffle_str, fold_ix, cross_ix, out_dir, num_chunks, num_genes):
    
    # initialize final h5
    final_h5_open = h5py.File(file_name, "w")
    
    seqs = []
    isms = []
    genes = []
    chrs = []
    starts = []
    ends = []
    utr_starts = []
    utr_ends = []
    strands = []
    
    for chunk_i in range(num_chunks) :
        chunk_h5_file = "%s/ism%s_f%dc%d_part%d.h5" % (out_dir, shuffle_str, fold_ix, cross_ix, chunk_i)
        chunk_h5_open = h5py.File(chunk_h5_file, "r")
        
        seqs.append(chunk_h5_open["seqs"][()])
        isms.append(chunk_h5_open["isms"][()])
        genes.append(chunk_h5_open["gene"][()])
        chrs.append(chunk_h5_open["chr"][()])
        starts.append(chunk_h5_open["start"][()])
        ends.append(chunk_h5_open["end"][()])
        utr_starts.append(chunk_h5_open["utr_start"][()])
        utr_ends.append(chunk_h5_open["utr_end"][()])
        strands.append(chunk_h5_open["strand"][()])
        
        chunk_h5_open.close()
    
    seqs = np.concatenate(seqs, axis=0)
    isms = np.concatenate(isms, axis=0)
    genes = np.concatenate(genes, axis=0)
    chrs = np.concatenate(chrs, axis=0)
    starts = np.concatenate(starts, axis=0)
    ends = np.concatenate(ends, axis=0)
    utr_starts = np.concatenate(utr_starts, axis=0)
    utr_ends = np.concatenate(utr_ends, axis=0)
    strands = np.concatenate(strands, axis=0)

    # store merged arrays
    final_h5_open.create_dataset("seqs", data=np.array(seqs, dtype="bool"))
    final_h5_open.create_dataset("isms", data=np.array(isms, dtype="float16"))
    final_h5_open.create_dataset("gene", data=np.array(genes, dtype="S"))
    final_h5_open.create_dataset("chr", data=np.array(chrs, dtype="S"))
    final_h5_open.create_dataset("start", data=np.array(starts))
    final_h5_open.create_dataset("end", data=np.array(ends))
    final_h5_open.create_dataset("utr_start", data=np.array(utr_starts))
    final_h5_open.create_dataset("utr_end", data=np.array(utr_ends))
    final_h5_open.create_dataset("strand", data=np.array(strands, dtype="S"))
    
    final_h5_open.close()
    
    return


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
