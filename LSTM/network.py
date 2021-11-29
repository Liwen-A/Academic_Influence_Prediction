import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import Parameter
from cupy.cuda import function
from pynvrtc.compiler import Program
from collections import namedtuple


class oLSTM(nn.Module):
    def __init__(self, ninp, nhid, nlayers, nout):
        super(oLSTM, self).__init__()
        self.rnn = nn.LSTM(ninp,nhid,nlayers)
        self.classifier = nn.Linear(nhid, nout)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'classifier' in name and 'weight' in name:
                nn.init.kaiming_normal_(param.data)

    def forward(self, input):
        output, (hn,cn) = self.rnn(input)
        output = self.classifier(output[-1])
        return output
