## Data Processing

This tutorial decribes how to process a .bigwig sequencing experiment into compressed .w5 format, merge replicates, generate QC metrics, and finally create TFRecord files containing binned coverage values suitable for training Borzoi models. We will exemplify this for the ENCODE K562 RNA-seq experiment [ENCSR000AEL](https://www.encodeproject.org/experiments/ENCSR000AEL/).

First, activate the conda environment and run the script 'download_dependencies.sh' to download required auxiliary files.
```sh
conda activate borzoi_py310
cd ~/borzoi/tutorials/latest/make_data
./download_dependencies.sh
```

Next, run the script 'download_bw.sh' to download sample ENCODE .bigwig files and arrange them in a folder structure.
```sh
./download_bw.sh
```

Then run script 'process_w5.sh' to generate compressed .w5 files (hdf5) from the input .bigwig files, merge the two replicates, and calculate basic QC metrics. This .sh script internally calls 'bw_h5.py' to generate .w5 files, 'w5_merge.py' to merge replicates, and 'w5_qc.py' to calculate QC metrics.
```sh
./process_w5.sh
```

Finally, run the Makefile to create genome-wide binned coverage tracks, stored as compressed TFRecords.
```sh
make
```

In this example, the Makefile creates 8 cross-validation folds of TFRecords with input sequences of length 393216 bp, generated with a genome-wide stride of 43691 bp. The output coverage tracks corresponding to each input sequence are not cropped in the latest version of Borzoi models. This results in 12288 coverage bins per 393kb sequence. The specific .w5 tracks to include in the TFRecord generation, and the scales and pooling transforms applied to the bins of each experiment, are given in the targets file 'targets_human.txt'. Below is a description of the columns in this file.

*targets_human.txt*:
- (unnamed) => integer index of each track (must start from 0 when training a new model).
- 'identifier' => unique identifier of each experiment (and strand).
- 'file' => local file path to .w5 file.
- 'clip' => hard clipping threshold to be applied to each bin, after soft-clipping.
- 'clip_soft' => soft clipping (squashing) threshold.
- 'scale' => scale value applied to each bp-level position before clipping.
- 'sum_stat' => type of bin-level pooling operation ('sum_sqrt' = sum and square-root).
- 'strand_pair' => integer index of the other stranded track of an experiment (same index as current row if unstranded).
- 'description' => text description of experiment.

*Notes*:
- See [here](https://github.com/calico/borzoi-paper/tree/main/data/training) for a description of the scripts called by the Makefile to create TFRecords.
- In the latest version of Borzoi models, a modified hg38 fasta genome is used in the Makefile where the allele with highest overall frequency (from gnomAD) is substituted at each position.
