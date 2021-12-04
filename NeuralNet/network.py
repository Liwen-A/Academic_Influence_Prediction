# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 01:11:38 2021

@author: Shreya
"""

import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNet(nn.Module):
    def __init__(self, ninp, nhid1, nhid2, nout):
        super(NeuralNet, self).__init__()
#        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(ninp, nhid1),
            nn.LeakyReLU(0.1),
            nn.Linear(nhid1, nhid2),
            nn.LeakyReLU(0.1),
            nn.Linear(nhid2, nout),
        )

    def forward(self, x):
#        x = self.flatten(x)
        output = self.linear_relu_stack(x)
        return output
class SparseNN(nn.Module):
    def __init__(self):
        super(SparseNN, self).__init__()
        self.network1 = NeuralNet(300, 1000, 100, 20) #neural net for word2vec feature
        self.network2 = NeuralNet(12, 30, 20, 10) #neural net for rest of features
        
        self.relu = nn.LeakyReLU(0.1)
        self.fc1 = nn.Linear(30, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc_out = nn.Linear(10, 1)
        
    def forward(self, x1, x2):
        x1 = self.relu(self.network1(x1))
        x2 = self.relu(self.network2(x2))

        
        x = torch.cat((x1, x2), 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc_out(x)
        return x