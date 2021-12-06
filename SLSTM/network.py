import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import Parameter
from cupy.cuda import function
from pynvrtc.compiler import Program
from collections import namedtuple


class oLSTM(nn.Module):
    def __init__(self, ninp_fc, nhid_fc, nout_fc, ninp_abs, nhid_abs, ninp, nhid, nlayers, batch_size, nout):
        super(oLSTM, self).__init__()
        self.rnn = nn.LSTM(ninp,nhid,nlayers)
        self.classifier = nn.Linear(nhid, nout)
        self.inp_linear = nn.Linear(ninp_fc, nhid_fc)
        self.hid_linear = nn.Linear(nhid_fc, nhid_fc)
        self.out_linear = nn.Linear(nhid_fc, nout_fc)
        self.inp_abs_linear = nn.Linear(ninp_abs, nhid_abs)
        self.hid_abs_linear = nn.Linear(nhid_abs, nhid_abs)
        self.out_abs_linear = nn.Linear(nhid_abs, nout_fc)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'classifier' in name and 'weight' in name:
                nn.init.kaiming_normal_(param.data)

    def forward(self, input_fc, input_abs, input):
        hid1   = self.relu(self.inp_linear(input_fc))
        hid2   = self.relu(self.hid_linear(hid1))
        out_fc = self.out_linear(hid2)
        hid1_abs = self.relu(self.inp_abs_linear(input_abs))
        hid2_abs = self.relu(self.hid_abs_linear(hid1_abs))
        out_abs  = self.out_abs_linear(hid2_abs)
        out_comb = torch.cat((out_fc, out_abs), 1)
        output, (hn,cn) = self.rnn(input, (torch.unsqueeze(out_comb, 0), torch.zeros_like(torch.unsqueeze(out_comb, 0))))
        output = self.classifier(output[-1])
        return output
