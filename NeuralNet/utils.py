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
    '''
    L = []
    for name in names:
        fp = os.path.join('CitationByYearData', name + ".csv")
        df = pd.read_csv(fp).dropna()
        L.append(df)
    df0 = pd.concat(L)
    '''
    
    w2vdf1 = pd.read_csv('../Academic_Influence_Prediction/CitationByYearData/w2vPhysicsNobel.csv').dropna()
    w2vdf2 = pd.read_csv('../Academic_Influence_Prediction/CitationByYearData/w2vChemNobel.csv').dropna()
    w2vdf3 = pd.read_csv('../Academic_Influence_Prediction/CitationByYearData/w2vMedNobel.csv').dropna()
    datadf1 = pd.read_csv('../Academic_Influence_Prediction/CitationByYearData/PhysicsNobel.csv').dropna()
    datadf2 = pd.read_csv('../Academic_Influence_Prediction/CitationByYearData/ChemNobel.csv').dropna()
    datadf3 = pd.read_csv('../Academic_Influence_Prediction/CitationByYearData/MedNobel.csv').dropna()
    w2vdf = pd.concat([w2vdf1, w2vdf2, w2vdf3], axis=0)
    datadf = pd.concat([datadf1, datadf2, datadf3], axis=0)

    xy = pd.concat([w2vdf, datadf], axis=1)
    xy = xy.drop_duplicates(subset='paperId', keep='first')
    # remove outliers
    q_low = xy["citationCount"].quantile(0.01)
    q_hi  = xy["citationCount"].quantile(0.99)

    xy = xy[(xy["citationCount"] < q_hi) & (xy["citationCount"] > q_low)]

    xy.to_csv('./xy.csv')
    datasize, dim = xy.shape


    xy = xy.sample(frac=1).reset_index(drop=True)
    citdf = xy.loc[:, 'citationCount']
    citationyeardf = xy.loc[:, 'year0_citation_count':'year9_citation_count']
    xy = xy.drop(columns=['paperId', 'title', 'abstract'])
    w2vdf = xy.loc[:, '0':'299'] * 100 # to increase importance of vector, but rescaling ruins the information


    xy=(xy-xy.min())/(xy.max()-xy.min())
    #citationyeardf = xy.loc[:, 'year0_citation_count':'year9_citation_count']
    refdf = xy.loc[:,'referenceCount']
    #citdf = xy.loc[:, 'citationCount']
    yeardf = xy.loc[:, 'year']


    #featuresdf = pd.concat([w2vdf, citationyeardf, refdf, yeardf], axis=1)
    featuresdf = pd.concat([citationyeardf, refdf, yeardf], axis=1)

    #features1df = pd.concat([citationyeardf, refdf, yeardf], axis=1)


    #train_data = df0[['referenceCount', 'citationCount']].to_numpy(dtype=np.float32)
    #train_labels = df0['influentialCitationCount'].to_numpy(dtype=np.float32)
    #data_size = len(train_labels)
    
    #train_data, test_data = torch.utils.data.random_split(train_data, [int(0.8*data_size),data_size - int(0.8*data_size)])
    #train_labels, test_labels = torch.utils.data.random_split(train_labels, [int(0.8*data_size),data_size - int(0.8*data_size)])    
    '''
    train_data = featuresdf[5000:].to_numpy(dtype=np.float32)
    train_labels = citdf[5000:].to_numpy(dtype=np.float32)

    test_data = featuresdf[:5000].to_numpy(dtype=np.float32)
    test_labels = citdf[:5000].to_numpy(dtype=np.float32)

    train_data = featuresdf[5000:].to_numpy(dtype=np.float32)
    train_labels = citdf[5000:].to_numpy(dtype=np.float32)
    '''
    train_data1 = w2vdf[3000:].to_numpy(dtype=np.float32)
    train_data2 = featuresdf[3000:].to_numpy(dtype=np.float32)
    train_labels = citdf[3000:].to_numpy(dtype=np.float32)

    test_data1 = w2vdf[:3000].to_numpy(dtype=np.float32)
    test_data2 = featuresdf[:3000].to_numpy(dtype=np.float32)
    test_labels = citdf[:3000].to_numpy(dtype=np.float32)    
    '''
    ## Train data:
    train_dataset = TensorDataset(Tensor(train_data).float(), Tensor(train_labels).float())
    trainloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_train)

    ## Test data
    test_dataset = TensorDataset(Tensor(test_data).float(), Tensor(test_labels).float())
    testloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_test)
    '''
    ## Train data:
    train_dataset = TensorDataset(Tensor(train_data1).float(),Tensor(train_data2).float(), Tensor(train_labels).float())
    trainloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_train)

    ## Test data
    test_dataset = TensorDataset(Tensor(test_data1).float(), Tensor(test_data2).float(),Tensor(test_labels).float())
    testloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_test)

    return trainloader, train_dataset, testloader, test_dataset