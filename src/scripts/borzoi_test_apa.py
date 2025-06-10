#!/usr/bin/env python
# Copyright 2021 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from optparse import OptionParser
import gc
import json
import os
import time
import sys

import numpy as np
import pandas as pd
import pyranges as pr

from baskerville import dataset
from baskerville import seqnn

"""
borzoi_test_apa.py

Measure accuracy at polyadenylation-level.
"""

################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] <params_file> <model_file> <data_dir> <exons_gff>"
    parser = OptionParser(usage)
    parser.add_option(
        "--head",
        dest="head_i",
        default=0,
        type="int",
        help="Parameters head [Default: %default]",
    )
    parser.add_option(
        "-o",
        dest="out_dir",
        default="teste_out",
        help="Output directory for predictions [Default: %default]",
    )
    parser.add_option(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Average the fwd and rc predictions [Default: %default]",
    )
    parser.add_option(
        "--shifts",
        dest="shifts",
        default="0",
        help="Ensemble prediction shifts [Default: %default]",
    )
    parser.add_option(
        "-t",
        dest="targets_file",
        default=None,
        type="str",
        help="File specifying target indexes and labels in table format",
    )
    parser.add_option(
        "--split",
        dest="split_label",
        default="test",
        help="Dataset split label for eg TFR pattern [Default: %default]",
    )
    parser.add_option(
        "--tfr",
        dest="tfr_pattern",
        default=None,
        help="TFR pattern string appended to data_dir/tfrecords for subsetting [Default: %default]",
    )
    parser.add_option(
        '--stat',
        dest='cov_stat',
        default='COVR',
        help='Coverage statistic to aggregate. [Default: %default]'
    )
    parser.add_option(
        '--utr3',
        dest='utr3',
        default=False, action='store_true',
        help='Only aggregate coverage over sites in the 3-prime UTR. [Default: %default]'
    )
    parser.add_option(
        "-u",
        dest="untransform_old",
        default=False,
        action="store_true",
        help="Untransform old models [Default: %default]",
    )
    (options, args) = parser.parse_args()

    if len(args) != 4:
        parser.error(
            "Must provide parameters, model, data directory, and APA annotation"
        )
    else:
        params_file = args[0]
        model_file = args[1]
        data_dir = args[2]
        apa_file = args[3]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    # parse shifts to integers
    options.shifts = [int(shift) for shift in options.shifts.split(",")]

    #######################################################
    # inputs

    # read targets
    if options.targets_file is None:
        options.targets_file = "%s/targets.txt" % data_dir
    targets_df = pd.read_csv(options.targets_file, index_col=0, sep="\t")

    # attach strand
    targets_strand = []
    for ti, identifier in enumerate(targets_df.identifier):
        if targets_df.index[ti] == targets_df.strand_pair.iloc[ti]:
            targets_strand.append(".")
        else:
            targets_strand.append(identifier[-1])
    targets_df["strand"] = targets_strand

    # collapse stranded
    strand_mask = targets_df.strand != "-"
    targets_strand_df = targets_df[strand_mask]

    # count targets
    num_targets = targets_df.shape[0]
    num_targets_strand = targets_strand_df.shape[0]

    # read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]
    params_train = params["train"]

    # set strand pairs
    if 'strand_pair' in targets_df.columns:
        orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
        targets_strand_pair = np.array([orig_new_index[ti] for ti in targets_df.strand_pair])
        params_model["strand_pair"] = [targets_strand_pair]

    # construct eval data
    eval_data = dataset.SeqDataset(
        data_dir,
        split_label=options.split_label,
        batch_size=params_train["batch_size"],
        mode="eval",
        tfr_pattern=options.tfr_pattern,
    )

    # initialize model
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file, options.head_i)
    seqnn_model.build_slice(targets_df.index)
    seqnn_model.build_ensemble(options.rc, options.shifts)

    #######################################################
    # sequence intervals

    # read data parameters
    with open("%s/statistics.json" % data_dir) as data_open:
        data_stats = json.load(data_open)
        crop_bp = data_stats["crop_bp"]
        pool_width = data_stats["pool_width"]

    # read sequence positions
    seqs_df = pd.read_csv(
        "%s/sequences.bed" % data_dir,
        sep="\t",
        names=["Chromosome", "Start", "End", "Name"],
    )
    seqs_df = seqs_df[seqs_df.Name == options.split_label]
    seqs_pr = pr.PyRanges(seqs_df)

    #######################################################
    # make APA BED (PolyADB)

    apa_df = pd.read_csv(apa_file, sep="\t", compression="gzip")

    # optionally filter for 3' UTR polyA sites only
    if options.utr3 :
        apa_df = apa_df.query("site_type == '3\\' most exon'").copy().reset_index(drop=True)
    else :
        apa_df = apa_df.query("site_type == '3\\' most exon' or site_type == 'Intron'").copy().reset_index(drop=True)

    apa_df["start_hg38"] = apa_df["position_hg38"]
    apa_df["end_hg38"] = apa_df["position_hg38"] + 1

    apa_df = apa_df.rename(
        columns={
            "chrom": "Chromosome",
            "start_hg38": "Start",
            "end_hg38": "End",
            "position_hg38": "cut_mode",
            "strand": "pas_strand",
        }
    )

    apa_pr = pr.PyRanges(
        apa_df[["Chromosome", "Start", "End", "pas_id", "cut_mode", "pas_strand"]]
    )
    
    # get strands
    pas_strand_dict = {}
    for _, row in apa_df.iterrows() :
        pas_strand_dict[row['pas_id']] = row['pas_strand']

    #######################################################
    # intersect APA sites w/ preds, targets

    # intersect seqs, APA sites
    seqs_apa_pr = seqs_pr.join(apa_pr)

    # hash preds/targets by pas_id
    apa_preds_dict = {}
    apa_targets_dict = {}

    si = 0
    for x, y in eval_data.dataset:
        # predict only if gene overlaps
        yh = None
        y = y.numpy()[..., targets_df.index]

        t0 = time.time()
        print("Sequence %d..." % si, end="", flush=True)
        for bsi in range(x.shape[0]):
            seq = seqs_df.iloc[si + bsi]

            cseqs_apa_df = seqs_apa_pr[seq.Chromosome].df
            if cseqs_apa_df.shape[0] == 0:
                # empty. no apa sites on this chromosome
                seq_apa_df = cseqs_apa_df
            else:
                seq_apa_df = cseqs_apa_df[cseqs_apa_df.Start == seq.Start]

            for _, seq_apa in seq_apa_df.iterrows():
                pas_id = seq_apa.pas_id
                pas_start = seq_apa.Start_b
                pas_end = seq_apa.End_b
                seq_start = seq_apa.Start
                cut_mode = seq_apa.cut_mode
                pas_strand = seq_apa.pas_strand

                # clip boundaries
                pas_seq_start = max(0, pas_start - seq_start)
                pas_seq_end = max(0, pas_end - seq_start)
                cut_seq_mode = max(0, cut_mode - seq_start)

                # requires >50% overlap
        
                bin_start = None
                bin_end = None

                # aggregate RNA-seq coverage
                if options.cov_stat == 'COVR' :
                    if pas_strand == '+' :
                        bin_end = int(np.round(pas_seq_start / pool_width)) + 1
                        if pool_width == 32 :
                            bin_start = bin_end - 3 - 1
                        else : #16
                            bin_start = bin_end - 8 - 1
                    else :
                        bin_start = int(np.round(pas_seq_end / pool_width))
                        if pool_width == 32 :
                            bin_end = bin_start + 3 + 1
                        else : #16
                            bin_end = bin_start + 8 + 1
                elif options.cov_stat == 'COVR3' :
                    if pas_strand == '+' :
                        bin_end = int(np.round(pas_seq_start / pool_width)) + 1
                        if pool_width == 32 :
                            bin_start = bin_end - 7 - 1
                        else : #16
                            bin_start = bin_end - 14 - 1
                    else :
                        bin_start = int(np.round(pas_seq_end / pool_width))
                        if pool_width == 32 :
                            bin_end = bin_start + 7 + 1
                        else : #16
                            bin_end = bin_start + 14 + 1
                elif options.cov_stat == 'PROP3' :
                    if pool_width == 32 :
                        bin_end = int(np.round(pas_seq_start / pool_width)) + 3
                        bin_start = bin_end - 5
                    else : #16
                        bin_end = int(np.round(pas_seq_start / pool_width)) + 5
                        bin_start = bin_end - 9

                # predict
                if yh is None:
                    yh = seqnn_model(x)

                # slice gene region
                yhb = yh[bsi, bin_start:bin_end].astype("float16")
                yb = y[bsi, bin_start:bin_end].astype("float16")

                if len(yb) > 0:
                    apa_preds_dict.setdefault(pas_id, []).append(yhb)
                    apa_targets_dict.setdefault(pas_id, []).append(yb)
                else:
                    print("(Warning: len(yb) <= 0)", flush=True)

        # advance sequence table index
        si += x.shape[0]
        print("DONE in %ds." % (time.time() - t0), flush=True)

        if si % 128 == 0:
            gc.collect()

    #######################################################
    # aggregate pA bin values into arrays

    apa_targets = []
    apa_preds = []
    pas_ids = np.array(sorted(apa_targets_dict.keys()))

    for pas_id in pas_ids:
        apa_preds_gi = np.concatenate(apa_preds_dict[pas_id], axis=0).astype("float32")
        apa_targets_gi = np.concatenate(apa_targets_dict[pas_id], axis=0).astype(
            "float32"
        )
        
        # slice strand
        if pas_strand_dict[pas_id] == "+":
            pas_strand_mask = (targets_df.strand != "-").to_numpy()
        else:
            pas_strand_mask = (targets_df.strand != "+").to_numpy()
        apa_preds_gi = apa_preds_gi[:, pas_strand_mask]
        apa_targets_gi = apa_targets_gi[:, pas_strand_mask]

        # untransform
        if options.untransform_old:
            apa_preds_gi = dataset.untransform_preds1(apa_preds_gi, targets_strand_df, unscale=True, unclip=False)
            apa_targets_gi = dataset.untransform_preds1(apa_targets_gi, targets_strand_df, unscale=True, unclip=False)
        else:
            apa_preds_gi = dataset.untransform_preds(apa_preds_gi, targets_strand_df, unscale=True, unclip=False)
            apa_targets_gi = dataset.untransform_preds(apa_targets_gi, targets_strand_df, unscale=True, unclip=False)

        # mean coverage
        apa_preds_gi = apa_preds_gi.mean(axis=0)
        apa_targets_gi = apa_targets_gi.mean(axis=0)

        apa_preds.append(apa_preds_gi)
        apa_targets.append(apa_targets_gi)

    apa_targets = np.array(apa_targets)
    apa_preds = np.array(apa_preds)

    # save numpy arrays with values
    np.save("%s/apa_targets.npy" % options.out_dir, apa_targets)
    np.save("%s/apa_preds.npy" % options.out_dir, apa_preds)

    # save values
    apa_targets_df = pd.DataFrame(apa_targets, index=pas_ids)
    apa_targets_df.to_csv(
        "%s/apa_targets.tsv.gz" % options.out_dir, sep="\t"
    )
    apa_preds_df = pd.DataFrame(apa_preds, index=pas_ids)
    apa_preds_df.to_csv("%s/apa_preds.tsv.gz" % options.out_dir, sep="\t")


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
