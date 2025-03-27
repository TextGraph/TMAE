# TMAE

## Introduction

This is the datasets and source code for "TMAE: Entropy-aware Masked Autoencoder for Low-cost Traffic Flow Map Inference".

The framework of TMAE is as below:
![tmae-model](https://github.com/user-attachments/assets/d5316372-fbe0-46c0-831a-6fa5a25b064b)

## Dataset

The data.zip file the experimental datasets which are created from the public datasets [TaxiBj](https://github.com/yoshall/UrbanFM/tree/master/data) and [ChengDu and XiAn](https://github.com/luimoli/RATFM/tree/master/data). Many thanks to the authors.

```
# Example of directory structure for datasets
XiAn
<your_root_path>/data/XiAn/train/Y.npy    # fine-grained traffic flow maps

<your_root_path>/data/XiAn/valid/Y.npy    # fine-grained traffic flow maps

<your_root_path>/data/XiAn/test/Y.npy    # fine-grained traffic flow maps

```

## Usage

```bash
cd tmae

# change datapath and channel for train
# change datapath and modelpath for test
```

1. train

```
python train.py
```

2. test

```
python test.py
```
