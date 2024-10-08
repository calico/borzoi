## Interpretation

This tutorial describes how to compute gradient saliency scores (sequence attributions) with respect to various statistics computed for a list of input genes specified in a .gtf file. This example uses the pre-trained, published Borzoi model to compute gradients. To download this model, run the script 'download_models.sh' in the 'borzoi' root folder.

First, to compute input gradients with respect to the log-sum of coverage across the exons of the target gene, run the script 'run_gradients_expr_CFHR2.sh'.
```sh
conda activate borzoi_py310
cd ~/borzoi/tutorials/legacy/interpret_sequence
./run_gradients_expr_CFHR2.sh
```

To compute input gradients with respect to the log-ratio of coverage immediately upstream and downstream of the distal polyA site of the target gene, run the script 'run_gradients_polya_CD99.sh'.
```sh
./run_gradients_polya_CD99.sh
```

To compute input gradients with respect to the log-ratio of coverage of an exon of the target gene relative to intronic coverage, run the script 'run_gradients_splice_GCFC2.sh'.
```sh
./run_gradients_splice_GCFC2.sh
```
Currently, the splicing gradient script chooses one exon at random to compute gradients for. While this approach was favorable for the specific analysis of the manuscript, we acknowledge that this is not particularly useful to users wanting to investigate an exon of their choice. We plan on updating this script soon to allow users to specify which exon to calculate gradients for.

*Notes*:
- The track scale, squashing exponentiation, and clip-soft threshold, are specific in the .py script arguments (flags: '--track_scale, '--track_transform', '--clip_soft'), and the values in the targets file are ignored. This means that the same data transformation parameters are applied to all tracks specified in the targets file. To calculate gradients for groups of tracks with different data transforms, separate these tracks into different targets files, and execute the gradient script on each group separately.
- The legacy data transforms are activated in all above .sh scripts with the flag '--untransform_old'.
