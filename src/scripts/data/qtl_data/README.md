## QTL data processing

The scripts in this folder are used to extract fine-mapped causal sQTLs, paQTLs and ipaQTLs from the results of the eQTL Catalogue, as well as construct distance- and expression-matched negative SNPs.<br/>

*Notes*: 
- The pipeline requires the GTEx v8 (median) TPM matrix, which can be downloaded [here](https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz).
<br/>

As a prerequisite to generating any of the QTL datasets, run the following scripts (in order):
1. download_finemap.py
2. download_sumstat.py
3. merge_finemapping_tables.py
4. make_expression_tables.py
<br/>

To prepare the sQTL dataset, run these scripts:
1. sqtl_make_positive_sets.py
2. sqtl_make_negative_sets.py
<br/>

To prepare the paQTL dataset, run these scripts:
1. paqtl_make_positive_sets.py
2. paqtl_make_negative_sets.py
<br/>

To prepare the ipaQTL dataset, run these scripts:
1. ipaqtl_make_positive_sets.py
2. ipaqtl_make_negative_sets.py
<br/>

Finally, to generate the QTL VCF files, run this script:
1. make_vcfs.py
<br/>
