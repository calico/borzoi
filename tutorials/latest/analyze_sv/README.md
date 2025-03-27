
# Shift augmentation for improved indel scoring in DNA sequence-based ML models
This repository contains example analyses related to indels, structural variants, and tandem repeats. The manuscript is available here:<br/>

"Shift augmentation for improved indel scoring in DNA sequence-based ML models" - biorXiv link.

Contact *drk (at) @calicolabs.com* or *anya (at) @calicolabs.com* for questions.

## Indel / structural variant effect visualization

Please follow the installation steps on the main page. This code depends on the [baskerville](https://github.com/calico/baskerville.git) library and on plotly.
Install plotly into the working environment:

```sh
pip install plotly
```

After you've installed baskerville, download the dependencies for SV visualization example, and run the example script:

```sh
bash download_dependencies_SV.sh
python analyze_indel.sh
```

This will plot one indel/SV provided in the .vcf format. The script currently only handles one variant per run, so make sure your .vcf contains one variant.
Interactive plots for each available GTEx tissue and across all GTEx tissues will be put in the specified output directory.

## Tandem repeat scoring

This script will analyze the effect of tandem repeats by reducing and extending the specified short tandem repeat in the reference genome, then performing linear 
regression over log2FC of the gene expression of interest. A tiny STR table (subset of the result obtained in [this paper](https://www.nature.com/articles/s41588-019-0521-9)) 
is provided in the data folder.

```sh
bash download_dependencies_STR.sh
python score_STR.sh
```
