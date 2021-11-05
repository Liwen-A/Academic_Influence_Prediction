# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 01:10:49 2021

@author: Shreya
"""

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os

def get_data(names, batch_train, batch_test):
    
    L = []
    for name in names:
        fp = os.path.join('Data', name + ".csv")
        df = pd.read_csv(fp)
        L.append(df)
    df0 = pd.concat(L)
    
    train_data = df0[['referenceCount', 'citationCount']].to_numpy(dtype=np.float32)
    train_labels = df0['influentialCitationCount'].to_numpy(dtype=np.float32)
    data_size = len(train_labels)
    
    train_data, test_data = torch.utils.data.random_split(train_data, [int(0.8*data_size),data_size - int(0.8*data_size)])
    train_labels, test_labels = torch.utils.data.random_split(train_labels, [int(0.8*data_size),data_size - int(0.8*data_size)])    

    ## Train data:
    train_dataset = TensorDataset(Tensor(train_data).float(), Tensor(train_labels).float())
    trainloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_train)

    ## Test data
    test_dataset = TensorDataset(Tensor(test_data).float(), Tensor(test_labels).float())
    testloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_test)

    return trainloader, train_dataset, testloader, test_dataset