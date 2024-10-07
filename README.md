<!---[![Build/Release Python Package](https://github.com/calico/github-template-python-library/actions/workflows/release-new-version.yml/badge.svg?branch=main)](https://github.com/calico/github-template-python-library/actions/workflows/release-new-version.yml)--->
<!---[![Python formatting and tests](https://github.com/calico/github-template-python-library/actions/workflows/run-tests-formatting.yml/badge.svg?branch=main)](https://github.com/calico/github-template-python-library/actions/workflows/run-tests-formatting.yml)--->
<!---[![Validate prettier formatting](https://github.com/calico/github-template-python-library/actions/workflows/check-prettier-formatting.yml/badge.svg?branch=main)](https://github.com/calico/github-template-python-library/actions/workflows/check-prettier-formatting.yml)--->

<img src="borzoi_logo.png" width="200" />

# Borzoi - Predicting RNA-seq from DNA Sequence
Code repository for Borzoi models, which are convolutional neural networks trained to predict RNA-seq coverage at 32bp resolution given 524kb input sequences. The model is described in the following bioRxiv preprint:<br/>

[https://www.biorxiv.org/content/10.1101/2023.08.30.555582v1](https://www.biorxiv.org/content/10.1101/2023.08.30.555582v1).

Borzoi was trained on a large set of RNA-seq experiments from ENCODE and GTEx, as well as re-processed versions of the original Enformer training data (including ChIP-seq and DNase data from ENCODE, ATAC-seq data from CATlas, and CAGE data from FANTOM5). Here is a list of trained-on experiments: [human](https://raw.githubusercontent.com/calico/borzoi/main/examples/targets_human.txt) / [mouse](https://raw.githubusercontent.com/calico/borzoi/main/examples/targets_mouse.txt).

The repository contains example usage code (including jupyter notebooks for predicting and visualizing genetic variants) as well as links for downloading model weights, training data, QTL benchmark tasks, etc.

Contact *drk (at) @calicolabs.com* or *jlinder (at) @calicolabs.com* for questions about the model or data.

## Installation
Borzoi depends on the [baskerville repository](https://github.com/calico/baskerville.git), which can be installed by issuing the following commands:
```sh
git clone https://github.com/calico/baskerville.git
cd baskerville
pip install -e .
```

Next, install the [borzoi repository](https://github.com/calico/borzoi.git) by issuing the following commands:
```sh
git clone https://github.com/calico/borzoi.git
cd borzoi
pip install -e .
```

To train new models, the [westminster repository](https://github.com/calico/westminster.git) is also required and can be installed with these commands:
```sh
git clone https://github.com/calico/westminster.git
cd westminster
pip install -e .
```

These repositories further depend on a number of python packages (which are automatically installed with borzoi). See **pyproject.toml** for a complete list. The most important version dependencies are:
- Python == 3.10
- Tensorflow == 2.15.x (see [https://www.tensorflow.org/install/pip](https://www.tensorflow.org/install/pip))

*Note*: The example notebooks require jupyter, which can be installed with `pip install notebook`.<br/>
A new conda environment can be created with `conda create -n borzoi_py310 python=3.10`.<br/>
Some of the scripts in this repository start multi-process jobs and require [slurm](https://slurm.schedmd.com/).

Finally, the code base relies on a number of environment variables. For convenience, these can be configured in the active conda environment with the 'env_vars.sh' script. First, open up 'env_vars.sh' in each repository folder and change the two lines of code at the top to your username and local path. Then, issue these commands:
```sh
cd borzoi
conda activate borzoi_py310
./env_vars.sh
cd ../baskerville
./env_vars.sh
cd ../westminster
./env_vars.sh
```

Alternatively, the environment variables can be set manually:
```sh
export BORZOI_DIR=/home/<user_path>/borzoi
export PATH=$BORZOI_DIR/src/scripts:$PATH
export PYTHONPATH=$BORZOI_DIR/src/scripts:$PYTHONPATH

export BASKERVILLE_DIR=/home/<user_path>/baskerville
export PATH=$BASKERVILLE_DIR/src/baskerville/scripts:$PATH
export PYTHONPATH=$BASKERVILLE_DIR/src/baskerville/scripts:$PYTHONPATH

export WESTMINSTER_DIR=/home/<user_path>/westminster
export PATH=$WESTMINSTER_DIR/src/westminster/scripts:$PATH
export PYTHONPATH=$WESTMINSTER_DIR/src/westminster/scripts:$PYTHONPATH

export BORZOI_CONDA=/home/<user>/anaconda3/etc/profile.d/conda.sh
export BORZOI_HG38=$BORZOI_DIR/examples/hg38
export BORZOI_MM10=$BORZOI_DIR/examples/mm10
```

*Note*: The *baskerville* and *westminster* variables are only required for data processing and model training.

### Model Availability
The model weights can be downloaded as .h5 files from the URLs below. We trained a total of 4 model replicates with identical train, validation and test splits (test = fold3, validation = fold4 from [sequences_human.bed.gz](https://github.com/calico/borzoi/blob/main/data/sequences_human.bed.gz)).

[Borzoi Replicate 0 (human)](https://storage.googleapis.com/seqnn-share/borzoi/f0/model0_best.h5) | [(mouse)](https://storage.googleapis.com/seqnn-share/borzoi/f0/model1_best.h5)<br/>
[Borzoi Replicate 1 (human)](https://storage.googleapis.com/seqnn-share/borzoi/f1/model0_best.h5) | [(mouse)](https://storage.googleapis.com/seqnn-share/borzoi/f1/model1_best.h5)<br/>
[Borzoi Replicate 2 (human)](https://storage.googleapis.com/seqnn-share/borzoi/f2/model0_best.h5) | [(mouse)](https://storage.googleapis.com/seqnn-share/borzoi/f2/model1_best.h5)<br/>
[Borzoi Replicate 3 (human)](https://storage.googleapis.com/seqnn-share/borzoi/f3/model0_best.h5) | [(mouse)](https://storage.googleapis.com/seqnn-share/borzoi/f3/model1_best.h5)<br/>

Users can run the script *download_models.sh* to download all model replicates and annotations into the 'examples/' folder.
```sh
cd borzoi
./download_models.sh
```

#### Mini Borzoi Models
We have trained a collection of (smaller) model instances on various subsets of data modalities (or on all data modalities but with architectural changes compared to the original architecture). For example, some models are trained only on RNA-seq data while others are trained on DNase-, ATAC- and RNA-seq. Similarly, some model instances are trained on human-only data while others are trained on human- and mouse data. The models were trained with either 2- or 4-fold cross-validation and are available at the following URL:

[Mini Borzoi Model Collection](https://storage.googleapis.com/seqnn-share/borzoi/mini/)<br/>

For example, here are the weights, targets, and parameter file of a model trained on K562 RNA-seq:

[Borzoi K562 RNA-seq Fold 0](https://storage.googleapis.com/seqnn-share/borzoi/mini/k562_rna/f0/model0_best.h5)<br/>
[Borzoi K562 RNA-seq Fold 1](https://storage.googleapis.com/seqnn-share/borzoi/mini/k562_rna/f1/model0_best.h5)<br/>
[Borzoi K562 RNA-seq Targets](https://storage.googleapis.com/seqnn-share/borzoi/mini/k562_rna/hg38/targets.txt)<br/>
[Borzoi K562 RNA-seq Parameters](https://storage.googleapis.com/seqnn-share/borzoi/mini/k562_rna/params.json)<br/>

### Data Availability
The training data for Borzoi can be downloaded from the following URL:

[Borzoi Training Data](https://storage.googleapis.com/borzoi-paper/data/)<br/>

*Note*: This data bucket is very large and thus set to "Requester Pays".

### QTL Availability
The curated e-/s-/pa-/ipaQTL benchmarking data can be downloaded from the following URLs:

[eQTL Data](https://storage.googleapis.com/borzoi-paper/qtl/eqtl/)<br/>
[sQTL Data](https://storage.googleapis.com/borzoi-paper/qtl/sqtl/)<br/>
[paQTL Data](https://storage.googleapis.com/borzoi-paper/qtl/paqtl/)<br/>
[ipaQTL Data](https://storage.googleapis.com/borzoi-paper/qtl/ipaqtl/)<br/>

### Paper Replication
To replicate the results presented in the paper, visit the [borzoi-paper repository](https://github.com/calico/borzoi-paper.git). This repository contains scripts for **training**, **evaluating**, and **analyzing** the published model, and for processing the **training data**.

### Tutorials
The following directories contain *minimal* tutorials regarding model training, variant scoring, and interpretation. The 'legacy' tutorials use data transformations that are similar to those used in the manuscript, while 'latest' use updated (and simpler) transformations. Note that these tutorials are only intended to showcase core functionality on sample data (such as processing an RNA-seq experiment, or training a simple model). For advanced analyses, we recommend studying the results presented in the manuscript (see [Paper Replication](https://github.com/calico/borzoi/tree/main?tab=readme-ov-file#paper-replication)).

- **Data Processing** [latest](https://github.com/calico/borzoi/tree/main/tutorials/latest/make_data) | [legacy](https://github.com/calico/borzoi/tree/main/tutorials/legacy/make_data)<br/>
- **Model Training** [latest](https://github.com/calico/borzoi/tree/main/tutorials/latest/train_model) | [legacy](https://github.com/calico/borzoi/tree/main/tutorials/legacy/train_model)<br/>
- **Variant Scoring** [latest](https://github.com/calico/borzoi/tree/main/tutorials/latest/score_variants) | [legacy](https://github.com/calico/borzoi/tree/main/tutorials/legacy/score_variants)<br/>
- **Sequence Interpretation** [latest](https://github.com/calico/borzoi/tree/main/tutorials/latest/interpret_sequence) | [legacy](https://github.com/calico/borzoi/tree/main/tutorials/legacy/interpret_sequence)<br/>

### Example Notebooks
The following notebooks contain example code for predicting and interpreting genetic variants.

[Notebook 1a: Interpret eQTL SNP (expression)](https://github.com/calico/borzoi/blob/main/examples/borzoi_example_eqtl_chr10_116952944_T_C.ipynb) [(fancy)](https://github.com/calico/borzoi/blob/main/examples/borzoi_example_eqtl_chr10_116952944_T_C_fancy.ipynb)<br/>
[Notebook 1b: Interpret paQTL SNP (polyadenylation)](https://github.com/calico/borzoi/blob/main/examples/borzoi_example_paqtl_chr1_236763042_A_G.ipynb) [(fancy)](https://github.com/calico/borzoi/blob/main/examples/borzoi_example_paqtl_chr1_236763042_A_G_fancy.ipynb)<br/>
[Notebook 1c: Interpret sQTL SNP (splicing)](https://github.com/calico/borzoi/blob/main/examples/borzoi_example_sqtl_chr9_135548708_G_C.ipynb)<br/>
[Notebook 1d: Interpret ipaQTL SNP (splicing and polya)](https://github.com/calico/borzoi/blob/main/examples/borzoi_example_ipaqtl_chr10_116664061_G_A.ipynb)<br/>
