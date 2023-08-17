<!---[![Build/Release Python Package](https://github.com/calico/github-template-python-library/actions/workflows/release-new-version.yml/badge.svg?branch=main)](https://github.com/calico/github-template-python-library/actions/workflows/release-new-version.yml)--->
<!---[![Python formatting and tests](https://github.com/calico/github-template-python-library/actions/workflows/run-tests-formatting.yml/badge.svg?branch=main)](https://github.com/calico/github-template-python-library/actions/workflows/run-tests-formatting.yml)--->
<!---[![Validate prettier formatting](https://github.com/calico/github-template-python-library/actions/workflows/check-prettier-formatting.yml/badge.svg?branch=main)](https://github.com/calico/github-template-python-library/actions/workflows/check-prettier-formatting.yml)--->

# Borzoi - Predicting RNA-seq from DNA Sequence
Code repository for training and using Borzoi models, which are convolutional neural networks trained to predict RNA-seq coverage at 32bp resolution given 524kb sequences as input. The model is described in the following bioRxiv preprint:<br/>

[https://biorxiv.org/content/todo.1.2.3](https://biorxiv.org/content/todo.1.2.3).

Borzoi was trained on a large set of RNA-seq experiments from ENCODE and GTEx, as well as re-processed versions of the original Enformer training data (including ChIP-seq and DNase coverage tracks from ENCODE, ATAC-seq data from CATlas, and CAGE data from FANTOM5). You can find a detailed list of trained-on (human) experiments [here](https://raw.githubusercontent.com/calico/borzoi/main/examples/targets_human.txt).

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

The baskerville and borzoi repositories further depend on a number of python packages (which are automatically installed with borzoi). See **setup.cfg** for a complete list of dependencies. The most important version dependencies are:
- Python == 3.9
- Tensorflow == 2.11.0

*Note*: The example notebooks require jupyter, which can be installed with `pip install notebook`.

### Model Availability
The model weights can be downloaded as .h5 files from the following URLs:

[Borzoi V2 Cross-fold 0](https://storage.googleapis.com/seqnn-share/borzoi/f0/model0_best.h5)<br/>
[Borzoi V2 Cross-fold 1](https://storage.googleapis.com/seqnn-share/borzoi/f1/model0_best.h5)<br/>
[Borzoi V2 Cross-fold 2](https://storage.googleapis.com/seqnn-share/borzoi/f2/model0_best.h5)<br/>
[Borzoi V2 Cross-fold 3](https://storage.googleapis.com/seqnn-share/borzoi/f3/model0_best.h5)<br/>

### Data Availability
The training data for Borzoi can be downloaded from the following URL:

[Borzoi V2 Training Data](https://storage.googleapis.com/seqnn-share/todo.todo)<br/>

*Note*: This data bucket is very large and thus set to "Requester Pays".

### QTL Availability
The curated e-/s-/pa-/ipaQTL benchmarking data can be downloaded from the following URLs:

[eQTL Data](https://storage.googleapis.com/seqnn-share/todo.todo)<br/>
[sQTL Data](https://storage.googleapis.com/seqnn-share/todo.todo)<br/>
[paQTL Data](https://storage.googleapis.com/seqnn-share/todo.todo)<br/>
[ipaQTL Data](https://storage.googleapis.com/seqnn-share/todo.todo)<br/>

### Example Notebooks
The following notebooks contain example code for predicting and interpreting genetic variants.

[Notebook 1a: Interpret eQTL SNP (expression)](https://github.com/calico/borzoi/blob/main/examples/borzoi_example_eqtl_chr10_116952944_T_C.ipynb)<br/>
[Notebook 1b: Interpret sQTL SNP (splicing)](https://github.com/calico/borzoi/blob/main/examples/borzoi_example_sqtl_chr9_135548708_G_C.ipynb)<br/>
[Notebook 1c: Interpret paQTL SNP (polyadenylation)](https://github.com/calico/borzoi/blob/main/examples/borzoi_example_paqtl_chr1_236763042_A_G.ipynb)<br/>
[Notebook 1d: Interpret ipaQTL SNP (splicing and polya)](https://github.com/calico/borzoi/blob/main/examples/borzoi_example_ipaqtl_chr10_116664061_G_A.ipynb)<br/>
