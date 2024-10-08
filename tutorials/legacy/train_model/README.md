## Model Training

This tutorial describes how to train smaller Borzoi models on the example RNA-seq experiment processed in the [make_data tutorial](https://github.com/calico/borzoi/tree/main/tutorials/legacy/make_data).

To train a 'Mini Borzoi' ensemble (~40M parameters, 2 cross-validation folds), run the script 'train_mini.sh'. The model parameters are specified in 'params_mini.json'. This model can be trained with a batch size of 2 on a 24GB NVIDIA Titan RTX or RTX4090 GPU.
```sh
conda activate borzoi_py310
cd ~/borzoi/tutorials/legacy/train_model
./train_mini.sh
```

Alternatively, to train an even smaller 'Micro Borzoi' ensemble (~5M parameters), run the script 'train_micro.sh'. This model can fit into the above GPU cards with a batch size of 4, which means the learning rate can be doubled and each epoch finished in half the time.
```sh
./train_micro.sh
```

*Notes*:
- See [here](https://github.com/calico/borzoi-paper/tree/main/model) for a description of the scripts called internally by the training .sh script.
- The **legacy** model crops the predicted tracks (see layer 'Cropping1D' in the parameters file). In this example, the input sequence has length 393216 bp, and the cropping layer removes 3072x 32 bp bins from each side, resulting in 6144 bins.
- In the **legacy** architecture, there is an extra/superfluous linear convolution applied in each 'unet_conv' layer (see the bool 'upsample_conv' in the parameters file). This additional convolution has since been removed.
