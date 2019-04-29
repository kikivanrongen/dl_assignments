################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_hidden = num_hidden

        sigma = 1/num_hidden

        self.W_gx = nn.Parameter(torch.Tensor(input_dim, num_hidden).normal_(0,sigma))
        self.W_gh = nn.Parameter(torch.Tensor(num_hidden, num_hidden).normal_(0,sigma))
        self.b_g = nn.Parameter(torch.zeros(num_hidden,))

        self.W_ix = nn.Parameter(torch.Tensor(input_dim, num_hidden).normal_(0,sigma))
        self.W_ih = nn.Parameter(torch.Tensor(num_hidden, num_hidden).normal_(0,sigma))
        self.b_i = nn.Parameter(torch.zeros(num_hidden,))

        self.W_fx = nn.Parameter(torch.Tensor(input_dim, num_hidden).normal_(0,sigma))
        self.W_fh = nn.Parameter(torch.Tensor(num_hidden, num_hidden).normal_(0,sigma))
        self.b_f = nn.Parameter(torch.zeros(num_hidden,))

        self.W_ox = nn.Parameter(torch.Tensor(input_dim, num_hidden).normal_(0,sigma))
        self.W_oh = nn.Parameter(torch.Tensor(num_hidden, num_hidden).normal_(0,sigma))
        self.b_o = nn.Parameter(torch.zeros(num_hidden,))

        self.W_ph = nn.Parameter(torch.Tensor(num_hidden, num_classes).normal_(0,sigma))
        self.b_p = nn.Parameter(torch.zeros(num_classes,))

    def forward(self, x):

        h_t = torch.zeros(self.batch_size, self.num_hidden)
        c_t = torch.zeros(self.batch_size, self.num_hidden)

        for i in range(self.seq_length):
            g_t = torch.tanh(torch.mm(x[:,i].unsqueeze(1), self.W_gx) + torch.mm(h_t, self.W_gh) + self.b_g)
            i_t = torch.sigmoid(torch.mm(x[:,i].unsqueeze(1), self.W_ix) + torch.mm(h_t, self.W_ih) + self.b_i)
            f_t = torch.sigmoid(torch.mm(x[:,i].unsqueeze(1), self.W_fx) + torch.mm(h_t, self.W_fh) + self.b_o)
            o_t = torch.sigmoid(torch.mm(x[:,i].unsqueeze(1), self.W_ox) + torch.mm(h_t, self.W_oh) + self.b_f)
            c_t = g_t * i_t + c_t * f_t
            h_t = torch.tanh(c_t) * o_t
            p_t = torch.mm(h_t, self.W_ph) + self.b_p

        return p_t
