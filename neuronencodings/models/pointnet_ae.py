#!/usr/bin/env python3
__doc__ = """
PyTorch PointNet Autoencoder Implementation

Nicholas Turner, 2018 <nturner@cs.princeton.edu>
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd


class PointNetAE(nn.Module):

    def __init__(self, n_pts, pt_dim=3,
                 mlp1_fs=[512,256,256],
                 bottle_fs=128,
                 mlp2_fs=[256,256,512],
                 bn=False, bn_decay=None, act=F.relu):
        """
        channels_in - dimension on input points
        channels_out - dimension of outputs
        bottle_fs - bottleneck dimension
        pt_fs - # internal point features
        """

        nn.Module.__init__(self)

        # Parameters
        self.n_pts = n_pts
        self.pt_dim = pt_dim
        self.bottle_fs = bottle_fs
        self.bn = bn
        self.act = act
        self.bn_decay = bn_decay #not used yet

        #scaling mlp2 units for pt_dim
        mlp2_fs = [fs*pt_dim for fs in mlp2_fs]

        # Layers
        self.mlp1 = ConvMLP(pt_dim, *mlp1_fs, bottle_fs,
                            act=self.act, bn=self.bn)
        self.maxpool = nn.MaxPool1d(n_pts)
        self.mlp2 = LinearMLP(bottle_fs, *mlp2_fs,
                              act=self.act, bn=self.bn)
        self.output = linear(mlp2_fs[-1],n_pts*pt_dim, 1) #no activation fn here

    def forward(self, x):
        #expect input of size (batch_size, n_pts, pt_dim)
        batch_size = x.size()[0]
        x = x.transpose(2,1)
        x = self.maxpool(self.mlp1(x))
        global_fs = x.view(batch_size,-1)
        x = self.output(self.mlp2(global_fs))
        return x.view(batch_size, self.n_pts, self.pt_dim), global_fs

    def forward_global(self, x):
        x = x.transpose(2,1)
        x = self.mlp1(x)
        return x.max(1) #? to test


class ConvMLP(nn.Module):

    def __init__(self, *layer_channels, bn=False, act=F.relu, bn_decay=None):

        assert len(layer_channels) > 1, "need more than one channel"
        self.num_layers = len(layer_channels)
        self.bn = bn
        self.act = act

        nn.Module.__init__(self)

        for i in range(1,self.num_layers):
            c_in, c_out = layer_channels[i-1], layer_channels[i]

            layer_name = "conv{}".format(i)
            self.add_module(layer_name, conv(c_in, c_out, 1))

            if self.bn:
                bn_name = "bn{}".format(i)
                self.add_module(bn_name, nn.BatchNorm1d(c_out))

    def forward(self, x):
        for i in range(1,self.num_layers):
            x = getattr(self,"conv{}".format(i))(x)
            x = getattr(self,"bn{}".format(i))(x) if self.bn else x
            x = self.act(x)
        return x


def conv(channels_in, channels_out, ks, bias=True):
    conv = nn.Conv1d(channels_in, channels_out, ks, bias=bias)

    #Init
    nn.init.kaiming_normal_(conv.weight)
    if bias:
        nn.init.constant_(conv.bias, 0)

    return conv


class LinearMLP(nn.Module):

    def __init__(self, *layer_channels, bn=False, act=F.relu, bn_decay=None):

        assert len(layer_channels) > 1, "need more than one channel"
        self.num_layers = len(layer_channels)
        self.bn = bn
        self.act = act

        nn.Module.__init__(self)

        for i in range(1,self.num_layers):
            c_in, c_out = layer_channels[i-1], layer_channels[i]

            layer_name = "layer{}".format(i)
            self.add_module(layer_name, linear(c_in, c_out, bias=(not bn)))

            if self.bn:
                bn_name = "bn{}".format(i)
                self.add_module(bn_name, nn.BatchNorm1d(c_out))

    def forward(self, x):
        for i in range(1,self.num_layers):
            x = getattr(self, "layer{}".format(i))(x)
            x = getattr(self, "bn{}".format(i))(x) if self.bn else x
            x = self.act(x)
        return x


def linear(c_in, c_out, bias=True):
    linear = nn.Linear(c_in, c_out, bias=bias)

    nn.init.kaiming_normal_(linear.weight)
    if bias:
        nn.init.constant_(linear.bias, 0)

    return linear
