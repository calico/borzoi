<!---[![Build/Release Python Package](https://github.com/calico/github-template-python-library/actions/workflows/release-new-version.yml/badge.svg?branch=main)](https://github.com/calico/github-template-python-library/actions/workflows/release-new-version.yml)--->
<!---[![Python formatting and tests](https://github.com/calico/github-template-python-library/actions/workflows/run-tests-formatting.yml/badge.svg?branch=main)](https://github.com/calico/github-template-python-library/actions/workflows/run-tests-formatting.yml)--->
<!---[![Validate prettier formatting](https://github.com/calico/github-template-python-library/actions/workflows/check-prettier-formatting.yml/badge.svg?branch=main)](https://github.com/calico/github-template-python-library/actions/workflows/check-prettier-formatting.yml)--->

<img src="borzoi_logo.png" width="200" />

# Borzoi - Predicting RNA-seq from DNA Sequence
Code repository for Borzoi models, which are convolutional neural networks trained to predict RNA-seq coverage at 32bp resolution given 524kb input sequences. The model is described in the following bioRxiv preprint:<br/>

[https://www.biorxiv.org/content/10.1101/2023.08.30.555582v1](https://www.biorxiv.org/content/10.1101/2023.08.30.555582v1).

Borzoi was trained on a large set of RNA-seq experiments from ENCODE and GTEx, as well as re-processed versions of the original Enformer training data (including ChIP-seq and DNase data from ENCODE, ATAC-seq data from CATlas, and CAGE data from FANTOM5). Click [here](https://raw.githubusercontent.com/calico/borzoi/main/examples/targets_human.txt) for a list of trained-on experiments.

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

These repositories further depend on a number of python packages (which are automatically installed with borzoi). See **setup.cfg** for a complete list. The most important version dependencies are:
- Python == 3.9
- Tensorflow == 2.12.x (see [https://www.tensorflow.org/install/pip](https://www.tensorflow.org/install/pip))

*Note*: The example notebooks require jupyter, which can be installed with `pip install notebook`.<br/>
A new conda environment can be created with `conda create -n borzoi_py39 python=3.9`.

### Model Availability
The model weights can be downloaded as .h5 files from the URLs below. We trained a total of 4 model replicates with identical train, validation and test splits (test = fold3, validation = fold4 from [sequences_human.bed.gz](https://github.com/calico/borzoi/blob/main/data/sequences_human.bed.gz)).

[Borzoi V2 Replicate 0](https://storage.googleapis.com/seqnn-share/borzoi/f0/model0_best.h5)<br/>
[Borzoi V2 Replicate 1](https://storage.googleapis.com/seqnn-share/borzoi/f1/model0_best.h5)<br/>
[Borzoi V2 Replicate 2](https://storage.googleapis.com/seqnn-share/borzoi/f2/model0_best.h5)<br/>
[Borzoi V2 Replicate 3](https://storage.googleapis.com/seqnn-share/borzoi/f3/model0_best.h5)<br/>

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

[Borzoi V2 Training Data](https://storage.googleapis.com/borzoi-paper/data/)<br/>

*Note*: This data bucket is very large and thus set to "Requester Pays".

### QTL Availability
The curated e-/s-/pa-/ipaQTL benchmarking data can be downloaded from the following URLs:

[eQTL Data](https://storage.googleapis.com/borzoi-paper/qtl/eqtl/)<br/>
[sQTL Data](https://storage.googleapis.com/borzoi-paper/qtl/sqtl/)<br/>
[paQTL Data](https://storage.googleapis.com/borzoi-paper/qtl/paqtl/)<br/>
[ipaQTL Data](https://storage.googleapis.com/borzoi-paper/qtl/ipaqtl/)<br/>

### Example Notebooks
The following notebooks contain example code for predicting and interpreting genetic variants.

[Notebook 1a: Interpret eQTL SNP (expression)](https://github.com/calico/borzoi/blob/main/examples/borzoi_example_eqtl_chr10_116952944_T_C.ipynb)<br/>
[Notebook 1b: Interpret sQTL SNP (splicing)](https://github.com/calico/borzoi/blob/main/examples/borzoi_example_sqtl_chr9_135548708_G_C.ipynb)<br/>
[Notebook 1c: Interpret paQTL SNP (polyadenylation)](https://github.com/calico/borzoi/blob/main/examples/borzoi_example_paqtl_chr1_236763042_A_G.ipynb)<br/>
[Notebook 1d: Interpret ipaQTL SNP (splicing and polya)](https://github.com/calico/borzoi/blob/main/examples/borzoi_example_ipaqtl_chr10_116664061_G_A.ipynb)<br/>
