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
            nn.ReLU(),
            nn.Linear(nhid1, nhid2),
            nn.ReLU(),
            nn.Linear(nhid2, nout),
        )

    def forward(self, x):
#        x = self.flatten(x)
        output = self.linear_relu_stack(x)
        return output