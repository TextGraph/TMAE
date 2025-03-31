# TMAE

## Introduction

This is the datasets and source code for "TMAE: Entropy-aware Masked Autoencoder for Low-cost Traffic Flow Map Inference".

The framework of TMAE is as below:
![tmae-model](https://github.com/user-attachments/assets/d5316372-fbe0-46c0-831a-6fa5a25b064b)

## Dataset

The datasets are from the public datasets [TaxiBj](https://github.com/yoshall/UrbanFM/tree/master/data) and [ChengDu and XiAn](https://github.com/luimoli/RATFM/tree/master/data). Many thanks to the authors.

```
# Example of Dataset Directory Structure
XiAn
<your_root_path>/data/XiAn/train/
                                X.npy    # coarse-grained traffic flow maps
                                Y.npy    # fine-grained traffic flow maps
<your_root_path>/data/XiAn/valid/
                                X.npy
                                Y.npy
<your_root_path>/data/XiAn/test/
                                X.npy
                                Y.npy
```

## Usage

### 1. Clone the repository

```bash
git clone https://github.com/TextGraph/TMAE.git
cd tmae
```

### 2. Dataset Preparation

Extract datasets into the `TMAE/tame/data` directory.

### 3. train

To run train.py with specified hyperparameters, use the following command:

```
python train.py #default P1

python train.py --data_path P1 --channel 1 --patch_size 4 --model mae_vit_base_patch4
python train.py --data_path ChengDu --channel 2 --patch_size 2 --model mae_vit_base_patch2
python train.py --data_path XiAn --channel 2 --patch_size 2 --model mae_vit_base_patch2

```

Explanation:

- epochs: Number of training epochs (default: 400).

- batch_size: Batch size per GPU (default: 16).

- model: Specifies the model to train (default: mae_vit_base_patch4).

- norm_pix_loss: normalized pixel loss (not included means it remains disabled).

- data_path: Dataset path (default: P1).

- channel: Number of channels (default: 1).

- output_dir: Output directory for saving results (default: ./output).

- device: Device for training/testing (default: cuda).

- seed: Random seed for reproducibility (default: 2017).

- weight_decay: Weight decay for optimization (default: 0.05).

- lr: Learning rate (default is None, so you need to specify it).

- blr: Base learning rate (default: 2e-4).

### 4. test

To run test.py with specified hyperparameters, use the following command:

```
python test.py #default P1
python test.py --model mae_vit_base_patch4 --data_path P1 --patch_size 4
python test.py --model mae_vit_base_patch2 --data_path ChengDu --patch_size 2
python test.py --model mae_vit_base_patch2 --data_path XiAn --patch_size 2
```

Explanation:

- batch_size: Batch size per GPU (default: 16).

- model: Specifies the model to test (default: mae_vit_base_patch4).

- norm_pix_loss: normalized pixel loss (not included means it remains disabled).

- data_path: Dataset path (default: P1).

- seed: Random seed for reproducibility (default: 2017).
