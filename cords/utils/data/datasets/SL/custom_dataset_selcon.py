import numpy as np
import pandas as pd
import os
import torch
import torchvision
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torchvision import transforms

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


## Custom PyTorch Dataset Class wrapper
class CustomDataset_SELCON(Dataset):
    def __init__(self, data, target, device=None, transform=None):       
        self.transform = transform
        self.data = torch.from_numpy(data.astype('float32'))#.to(device)
        self.targets = torch.from_numpy(target)#.to(device)
        # if device is not None:
        #     # Push the entire data to given device, eg: cuda:0
        #     self.data = torch.from_numpy(data.astype('float32'))#.to(device)
        #     self.targets = torch.from_numpy(target)#.to(device)
        # else:
        #     self.data = data#.astype('float32')
        #     self.targets = target

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()        
        sample_data = self.data[idx]
        label = self.targets[idx]
        if self.transform is not None:
            sample_data = self.transform(sample_data)
        return (sample_data, label) #.astype('float32')

class CustomDataset_WithId_SELCON(Dataset):
    def __init__(self, data, target, transform=None):       
        self.transform = transform
        self.data = data #.astype('float32')
        self.targets = target
        self.X = self.data
        self.Y = self.targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  
        sample_data = self.data[idx]
        label = self.targets[idx]
        if self.transform is not None:
            sample_data = self.transform(sample_data)
        return sample_data, label,idx #.astype('float32')



class SubsetDataset_WithId_SELCON(CustomDataset_WithId_SELCON):
    def __init__(self, dataset, idxs, transform=None):
        super().__init__(dataset.data, dataset.targets, transform=transform)
        self.idxs = idxs
    
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, idx):
        new_idx = self.idxs[idx].tolist()
        data, targets, _ = super().__getitem__(new_idx)
        return (data, targets, idx)
