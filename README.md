# TMAE

## Introduction

This is the datasets and source code for "TMAE: Entropy-aware Masked Autoencoder for Low-cost Traffic Flow Map Inference".

The framework of TMAE is as below:
![tmae-model](https://github.com/user-attachments/assets/d5316372-fbe0-46c0-831a-6fa5a25b064b)

## Dataset

The datasets are created from the public datasets [TaxiBj](https://github.com/yoshall/UrbanFM/tree/master/data) and [ChengDu and XiAn](https://github.com/luimoli/RATFM/tree/master/data). Many thanks to the authors.

```
# Example directory structure of datasets
XiAn
<your_root_path>/data/XiAn/train/Y.npy    # fine-grained traffic flow maps
<your_root_path>/data/XiAn/valid/Y.npy    # fine-grained traffic flow maps
<your_root_path>/data/XiAn/test/Y.npy     # fine-grained traffic flow maps
```

## Usage

### 1. Clone the repository

```bash
git clone git@github.com:TextGraph/TMAE.git
cd tmae
```

### 2. Dataset Preparation

Extract datasets into the `TMAE/tame/data` directory.

### 3. train

To run train.py with specified hyperparameters, use the following command:

```
python train.py #default P1

python train.py --epochs 400 --batch_size 16 --model mae_vit_base_patch4 --norm_pix_loss \
--data_path P1 --channel 1 --output_dir ./output --device cuda --seed 2017 \
--weight_decay 0.05 --lr 0.001 --blr 2e-4
```

Explanation:

- epochs:Number of training epochs (default: 400).

- batch_size:Batch size per GPU (default: 16).

- model:Specifies the model to train (default: mae_vit_base_patch4).

- norm_pix_loss:normalized pixel loss (not included means it remains disabled).

- data_path:Dataset path (default: P1).

- channel:Number of channels (default: 1).

- output_dir:Output directory for saving results (default: ./output).

- device:Device for training/testing (default: cuda).

- seed:Random seed for reproducibility (default: 2017).

- weight_decay:Weight decay for optimization (default: 0.05).

- lr:Learning rate (default is None, so you need to specify it).

- blr:Base learning rate (default: 2e-4).

### 4. test

To run test.py with specified hyperparameters, use the following command:

```

python test.py #default P1
python test.py --batch_size 16 --model mae_vit_base_patch4 --norm_pix_loss \
--data_path P1 --device cuda --seed 2017
```

Explanation:

- batch_size:Batch size per GPU (default: 16).

- model:Specifies the model to train (default: mae_vit_base_patch4).

- norm_pix_loss:normalized pixel loss (not included means it remains disabled).

- data_path:Dataset path (default: P1).

- seed:Random seed for reproducibility (default: 2017).
