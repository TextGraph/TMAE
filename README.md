# TMAE

## Introduction

This is the source code for "TMAE: Entropy-aware Masked Autoencoder for Low-cost Traffic Flow Map Inference".

The framework of TMAE is as below:
![tmae-model](https://github.com/user-attachments/assets/d5316372-fbe0-46c0-831a-6fa5a25b064b)

## Dataset

The data.zip file is from the public datasets [TaxiBj](https://github.com/yoshall/UrbanFM/tree/master/data),[ChengDu and XiAn](https://github.com/luimoli/RATFM/tree/master/data). Many thanks to the authors.

```
# Example of directory structure for datasets
XiAn
<your_root_path>/data/XiAn/train/
                                X.npy    # coarse-grained traffic flow maps
                                Y.npy    # fine-grained traffic flow maps
                                ext.npy  # external factor vector
<your_root_path>/data/XiAn/valid/
                                X.npy    # coarse-grained traffic flow maps
                                Y.npy    # fine-grained traffic flow maps
                                ext.npy  # external factor vector
<your_root_path>/data/XiAn/test/
                                X.npy    # coarse-grained traffic flow maps
                                Y.npy    # fine-grained traffic flow maps
                                ext.npy  # external factor vector
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
