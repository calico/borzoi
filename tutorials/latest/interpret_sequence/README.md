## Interpretation

This tutorial describes how to compute gradient saliency scores (sequence attributions) with respect to various statistics computed for a list of input genes specified in a .gtf file. This example relies on the Mini Borzoi model trained on sample K562 RNA-seq data from the [train_model tutorial](https://github.com/calico/borzoi/tree/main/tutorials/latest/train_model), which clearly is a significantly weaker model than the pre-trained, published Borzoi model.

To compute input gradients with respect to the log-sum of coverage across the exons of the example gene HBE1, run the script 'run_gradients_expr_HBE1.sh'.
```sh
conda activate borzoi_py310
cd ~/borzoi/tutorials/latest/interpret_sequence
./run_gradients_expr_HBE1.sh
```

*Notes*:
- The track scale, squashing exponentiation, and clip-soft threshold, are specific in the .py script arguments (flags: '--track_scale, '--track_transform', '--clip_soft'), and the values in the targets file are ignored. This means that the same data transformation parameters are applied to all tracks specified in the targets file. To calculate gradients for groups of tracks with different data transforms, separate these tracks into different targets files, and execute the gradient script on each group separately.
