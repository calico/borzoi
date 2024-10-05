## Model Training

This tutorial describes how to train smaller Borzoi models on the example RNA-seq experiment processed in the [make_data tutorial](https://github.com/calico/borzoi/tree/main/tutorials/latest/make_data).

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
- Rather than cropping the output predictions before applying the training loss, in the latest version of Borzoi models a smooth position-specific loss weight is applied that penalizes prediction errors less at the left/right boundaries.
