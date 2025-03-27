from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import math
import torch

from matplotlib import pyplot as plt
class Dataset(Dataset):
    def __init__(self,path,mode='train',channel=1):
        self.X=None
        self.Y=None
        # self.mask_weights=None
        self.ext=None
        self.initialize(path,mode,channel)
        
    def initialize(self,path,mode,channel):
        datapath = os.path.join(path, mode)
        cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        # Tensor = torch.FloatTensor
        self.channel = channel
        if self.channel == 2:
            #---XIAN or CHENGDU
            self.X = Tensor(np.load(os.path.join(datapath, 'X.npy')))
            self.Y = Tensor(np.load(os.path.join(datapath, 'Y.npy')))
        elif self.channel == 1:
            #--BEIJING---
            self.X = Tensor(np.expand_dims(np.load(os.path.join(datapath, 'X.npy')), 1))
            self.Y = Tensor(np.expand_dims(np.load(os.path.join(datapath, 'Y.npy')), 1))
        self.ext = Tensor(np.load(os.path.join(datapath, 'ext.npy')))
        assert len(self.X) == len(self.Y)
        print('# {} samples: {}'.format(mode, len(self.X)))
    def __getitem__(self,index):
        return (self.X[index],self.Y[index],self.ext[index])
    def __len__(self):
        return self.X.size(0)
