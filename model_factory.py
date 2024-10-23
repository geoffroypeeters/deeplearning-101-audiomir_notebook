#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: model_module.py
Description: factory to create pytorch DL-models from *.yaml configuration file
Author: geoffroy.peeters@telecom-paris.fr
"""


import torch
import torch.nn as nn
import numpy as np
from model_module import SincConv_fast, ConvNeXtBlock, TemporalConvNet, ResidualBlock, depthwise_separable_conv





# Code: https://github.com/furkanyesiler/move
def f_autopool_weights(data, autopool_param):
    """
    Calculating the autopool weights for a given tensor
    :param data: tensor for calculating the softmax weights with autopool
    :return: softmax weights with autopool

    see https://arxiv.org/pdf/1804.10070
    alpha=0: unweighted mean
    alpha=1: softmax
    alpha=inf: max-pooling
    """
    # --- x: (batch, 256, 1, T)
    x = data * autopool_param
    # --- max_values: (batch, 256, 1, 1)
    max_values = torch.max(x, dim=3, keepdim=True).values
    # --- softmax (batch, 256, 1, T)
    softmax = torch.exp(x - max_values)
    # --- weights (batch, 256, 1, T)
    weights = softmax / torch.sum(softmax, dim=3, keepdim=True)
    return weights




def f_get_next_size(in_L, k, s):
    """ gives resulting output size of a convolution on a vector of len in_L with a kernel of size k and stride s """
    return int(np.floor( (in_L-k)/s+1 ))


def f_get_activation(activation_type):
    """ return the corresponding activation class """
    if activation_type=='Sigmoid': 
        activation = nn.Sigmoid()
    elif activation_type=='Softmax': 
        activation = nn.Softmax()
    elif activation_type=='ReLU': 
        activation = nn.ReLU()
    elif activation_type=='LeakyReLU': 
        activation = nn.LeakyReLU()
    elif activation_type=='PReLU':
        activation = nn.PReLU()
    return activation


class nnAbs(nn.Module):
    """ encapsultate *.abs as an object """
    def __init__(self):
        super().__init__()
    def forward(self, X):
        return torch.abs(X)

class nnMean(nn.Module):
    """ encapsultate *.mean as an object """
    def __init__(self, dim, keepdim):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
    def forward(self, X):
        return torch.mean(X, self.dim, self.keepdim)

class nnMax(nn.Module):
    """ encapsultate *.mean  as an object """
    def __init__(self, dim, keepdim):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
    def forward(self, X):
        out, _ = torch.max(X, self.dim, self.keepdim)
        return out

class nnSqueeze(nn.Module):
    """ encapsultate .squeeze as an object """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, X):
        out = X.squeeze(dim=self.dim)
        return out

class nnUnSqueeze(nn.Module):
    """ encapsultate .unsqueeze as an object """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, X):
        out = X.unsqueeze(dim=self.dim)
        return out

class nnPermute(nn.Module):
    """ encapsultate *.permute as an object """
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    def forward(self, X):
        return X.permute(self.shape)

class nnBatchNorm1dT(nn.Module):
    """ perform BatchNorm1d over transposed vector (in case input format is (B,T,C) """
    def __init__(self, num_features):
        super().__init__()
        self.module = nn.BatchNorm1d(num_features)
    def forward(self, X):
        return self.module(X.permute(0,2,1)).permute(0,2,1)

class nnDoubleChannel(nn.Module):
    """ encapsultate do-nothing as an object """
    def __init__(self):
        super().__init__()
    def forward(self, X):
        return X

class nnStoreAs(nn.Module):
    """ ??? """
    def __init__(self, key):
        super().__init__()
        self.key = key
    def __str__(self):
        return f"nnStoreAs(key={self.key})"
    def forward(self, X):
        return X


class nnCatWith(nn.Module):
    """ ??? """
    def __init__(self, key):
        super().__init__()
        self.key = key
    def __str__(self):
        return f"nnCatWith(key={self.key})"
    def forward(self, X):
        return X
 

class nnLSTMlast(nn.Module):
    """ ??? """
    def __init__(self, param):
        super().__init__()
        self.module = nn.LSTM(input_size=param.input_size, hidden_size=param.hidden_size, num_layers=param.num_layers, batch_first=True)
    def forward(self, X):
        # --- output: (batch_size, sequence_length, H_out)
        # --- h_n: (num_layers, batch_size, H_out) 
        # --- c_n: (num_layers, batch_size, H_out) 
        output, (h_n, c_n)  = self.module(X)
        return h_n[-1,:,:]


class nnLSTMall(nn.Module):
    """ ??? """
    def __init__(self, param):
        super().__init__()
        self.module = nn.LSTM(input_size=param.input_size, hidden_size=int(param.hidden_size/2), num_layers=param.num_layers, batch_first=True, bidirectional=True)
    def forward(self, X):
        # --- output: (batch_size, sequence_length, H_out)
        # --- h_n: (num_layers, batch_size, H_out) 
        # --- c_n: (num_layers, batch_size, H_out) 
        output, (h_n, c_n)  = self.module(X)
        return output


class nnSoftmaxWeight(nn.Module):
    """ 
    Perform attention weighing based on softmax with channel splitting
    Code from https://github.com/furkanyesiler/move 
    """
    def __init__(self, nb_channel):
        super().__init__()
        self.nb_channel = nb_channel
    def forward(self, X):
        weights = torch.nn.functional.softmax(X[:, int(self.nb_channel/2):], dim=3)
        X = torch.sum(X[:, :int(self.nb_channel/2)] * weights, dim=3, keepdim=True)
        return X

class nnAutoPoolWeight(nn.Module):
    """ 
    Perform attention weighing based on auto-pool (instead of softmax)
    Code from https://github.com/furkanyesiler/move 
    """
    def __init__(self):
        super().__init__()
        self.autopool_param = nn.Parameter(torch.tensor(0.).float())
    def forward(self, X):
        weights = f_autopool_weights(X, self.autopool_param)
        X = torch.sum(X * weights, dim=3, keepdim=True)
        return X

class nnAutoPoolWeightSplit(nn.Module):
    """ 
    Perform attention weighing based on auto-pool (instead of softmax) with channel splitting
    Code from https://github.com/furkanyesiler/move 
    """
    def __init__(self, nb_channel):
        super().__init__()
        self.autopool_param = nn.Parameter(torch.tensor(0.).float())
        self.nb_channel = nb_channel
    def forward(self, X):
        weights = f_autopool_weights(X[:, int(self.nb_channel/2):], self.autopool_param)
        X = torch.sum(X[:, :int(self.nb_channel/2):] * weights, dim=3, keepdim=True)
        return X



def f_parse_component(module_type, param, current_input_dim):
    """ 
    Parse module_type and param parameters to create a new NN layers, return the new dimensions of the tensor

    Parameters
    ----------
    module_type: str
        type of the module to be added (such as 'LayerNorm', 'BatchNorm1d', ...)
    param: dictionary
        the parameters for the given module
    current_input_dim: array of int
        dimension of the inputs before adding this specific layer

    Returns
    -------
    module
        the newly created nn.Module
    current_input_dim: : array of int
        dimension of the outputs aftere adding this specific layer
   """

    # --- FC:       B, D
    # --- Conv1D:   B, C, T
    # --- Conv2D:   B, C, H=Freq, W=Time

    #print(module_type, param, current_input_dim)
        
    if module_type=='LayerNorm':
        if param.normalized_shape==-1:
            param.normalized_shape = current_input_dim[1:] # --- B, C, T
        module = nn.LayerNorm(normalized_shape=param.normalized_shape)

    elif module_type=='BatchNorm1d':
        if param.num_features==-1:
            param.num_features = current_input_dim[-1] # --- (B, D) or (B, T, D)
        if 'affine' not in param.keys():
            param.affine = True
        module = nn.BatchNorm1d(num_features=param.num_features, affine=param.affine)

    elif module_type=='BatchNorm1dT':
        if param.num_features==-1:
            param.num_features = current_input_dim[-1] # --- (B, D) or (B, T, D)
        module = nnBatchNorm1dT(num_features=param.num_features)

    elif module_type=='BatchNorm2d':
        if param.num_features==-1:
            param.num_features = current_input_dim[1] # --- (B, C, H, W)
        module = nn.BatchNorm2d(num_features=param.num_features)

    elif module_type=='SincNet': # --- B, C, T
        if param.in_channels==-1:
            param.in_channels=current_input_dim[1]
        module = SincConv_fast(in_channels=param.in_channels, out_channels=param.out_channels, kernel_size=param.kernel_size, stride=param.stride, sample_rate=param.sr_hz)
        current_input_dim[1] = param.out_channels
        current_input_dim[2] = f_get_next_size(current_input_dim[2], param.kernel_size, param.stride)

    elif module_type=='LSTMlast':
        if param.input_size==-1:
            param.input_size=current_input_dim[2]
        module = nnLSTMlast(param)
        current_input_dim =  [current_input_dim[0], param.hidden_size]

    elif module_type=='LSTMall':
        if param.input_size==-1:
            param.input_size=current_input_dim[2]
        module = nnLSTMall(param)
        current_input_dim =  [current_input_dim[0], current_input_dim[1], param.hidden_size]

    elif module_type=='Conv1d': # --- B, C, T
        if param.in_channels==-1:
            param.in_channels=current_input_dim[1]
        if 'padding' not in param.keys():
            param.padding = 'valid'
        module = nn.Conv1d(in_channels=param.in_channels, out_channels=param.out_channels, kernel_size=param.kernel_size, stride=param.stride, padding=param.padding)
        current_input_dim[1] = param.out_channels
        if param.padding != 'same':
            current_input_dim[2] = f_get_next_size(current_input_dim[2], param.kernel_size, param.stride)

    elif module_type=='Conv1dTCN': # --- B, C, T
        if param.in_channels==-1:
            param.in_channels=current_input_dim[1]
        module = TemporalConvNet(num_inputs=param.in_channels, num_channels=param.num_channels)
        current_input_dim[1] = param.num_channels[-1]
        
    elif module_type=='Conv2d': # --- B, C, H=Freq, W=Time
        if param.in_channels==-1:
            param.in_channels=current_input_dim[1]
        if 'padding' not in param.keys():
            param.padding = 'valid'
        module = nn.Conv2d(in_channels=param.in_channels, out_channels=param.out_channels, kernel_size=param.kernel_size, stride=param.stride, padding=param.padding)
        current_input_dim[1] = param.out_channels
        if param.padding != 'same':
            current_input_dim[2] = f_get_next_size(current_input_dim[2], param.kernel_size[0], param.stride[0])
            current_input_dim[3] = f_get_next_size(current_input_dim[3], param.kernel_size[1], param.stride[1])

    elif module_type=='Conv2dDS': # --- B, C, H=Freq, W=Time
        if param.in_channels==-1:
            param.in_channels=current_input_dim[1]
        if 'padding' not in param.keys():
            param.padding = 'valid'
        module = depthwise_separable_conv(nin=param.in_channels, kernels_per_layer=1, nout=param.out_channels, kernel_size=param.kernel_size, padding=param.padding)
        current_input_dim[1] = param.out_channels
        if param.padding != 'same':
            current_input_dim[2] = f_get_next_size(current_input_dim[2], param.kernel_size[0], param.stride[0])
            current_input_dim[3] = f_get_next_size(current_input_dim[3], param.kernel_size[1], param.stride[1])

    elif module_type=='Conv2dRes': # --- B, C, H=Freq, W=Time
        if param.in_channels==-1:
            param.in_channels=current_input_dim[1]
        if param.padding != 'same':
            print(f'only work for padding=same, got {param}')
        if param.stride != 1:
            print(f'only work for stride=1, got {param}')
        module = ResidualBlock(in_channels=param.in_channels, out_channels=param.out_channels, kernel_size = param.kernel_size, stride = 1)
        current_input_dim[1] = param.out_channels
    
    elif module_type=='Conv2dNext': # --- B, C, H=Freq, W=Time
        if param.in_channels==-1:
            param.in_channels = current_input_dim[1]
        if param.out_channels != param.in_channels:
            print('param.out_channels should = param.in_channels because of residual connection')
        if param.padding != 'same':
            print(f'only work for padding=same, got {param}')
        if param.stride != 1:
            print(f'only work for stride=1, got {param}')
        module = ConvNeXtBlock(in_channels=param.in_channels, out_channels=param.out_channels, kernel_size=7, drop_path=0.0)
        current_input_dim[1] = param.out_channels

    elif module_type=='ConvTranspose2d':
        if param.in_channels==-1:
            param.in_channels = current_input_dim[1] 
        module = nn.ConvTranspose2d(in_channels=param.in_channels, out_channels=param.out_channels, kernel_size=param.kernel_size, stride=param.stride)
        current_input_dim[1] = param.out_channels
        current_input_dim[2] *= param.stride[0]
        current_input_dim[3] *= param.stride[1]

    elif module_type=='MaxPool1d':  # --- B, C, T
        if 'stride' not in param.keys():
            param.stride = param.kernel_size
        module = nn.MaxPool1d(kernel_size=param.kernel_size, stride=param.stride)
        current_input_dim[2] = int(np.floor(current_input_dim[2]/param.kernel_size)) 

    elif module_type=='MaxPool2d': # --- B, C, H=Freq, W=Time
        if 'stride' not in param.keys():
            param.stride = param.kernel_size
        module = nn.MaxPool2d(kernel_size=param.kernel_size, stride=param.stride)
        current_input_dim[2] = int(np.floor(current_input_dim[2]/param.kernel_size[0]))
        current_input_dim[3] = int(np.floor(current_input_dim[3]/param.kernel_size[1]))

    elif module_type=='Linear': 
        if param.in_features==-1:
            param.in_features = current_input_dim[-1] # --- (B, D) or (B, T, D)
        module = nn.Linear(in_features=param.in_features , out_features=param.out_features)
        current_input_dim[-1] = param.out_features

    elif module_type=='Activation':   
        module = f_get_activation(param)

    elif module_type=='Dropout':   
        module = nn.Dropout(param.p)

    elif module_type=='Flatten':
        module = nn.Flatten(param.start_dim)
        current_input_dim =  [current_input_dim[0], current_input_dim[1]*current_input_dim[2]]

    elif module_type=='Squeeze':
        module = nnSqueeze(param.dim)
        current_input_dim = [c for idx,c in enumerate(current_input_dim) if idx not in param.dim]
        
    elif module_type=='UnSqueeze':
        module = nnUnSqueeze(param.dim)
        current_input_dim.insert(param.dim, 1)

    elif module_type=='Permute':
        module = nnPermute(param.shape)
        current_input_dim =  [current_input_dim[s] for s in param.shape]

    elif module_type=='Mean':
        if 'keepdim' not in param.keys(): param.keepdim = False
        module = nnMean(param.dim, param.keepdim)
        if param.keepdim is False:
            current_input_dim =  [current_input_dim[0], current_input_dim[1]]

    elif module_type=='Max':
        if 'keepdim' not in param.keys(): param.keepdim = False
        module = nnMax(param.dim, param.keepdim)
        if param.keepdim is False:
            current_input_dim =  [current_input_dim[0], current_input_dim[1]]

    elif module_type=='AutoPoolWeight':
        module = nnAutoPoolWeight()
        current_input_dim[3] = 1

    elif module_type=='AutoPoolWeightSplit':
        module = nnAutoPoolWeightSplit(current_input_dim[1])
        current_input_dim[1] = int(current_input_dim[1]/2)
        current_input_dim[3] = 1

    elif module_type=='SoftmaxWeight':
        module = nnSoftmaxWeight(current_input_dim[1])
        current_input_dim[1] = int(current_input_dim[1]/2)
        current_input_dim[3] = 1

    elif module_type=='AbsLayer':
        module = nnAbs()

    elif module_type=='DoubleChannel': # --- for U-Net
        module = nnDoubleChannel()
        current_input_dim[1] *= 2

    elif module_type=='StoreAs':
        module = nnStoreAs(param)
        
    elif module_type=='CatWith':
        module = nnCatWith(param)
    
    else:
        print(f'UNKNOWN module "{module_type}"')

    return module, current_input_dim




class NetModel(nn.Module):
    """
    Generic class for neural-network models based on the f_parse_component of .yaml file
    """
    def __init__(self, config, current_input_dim):
        super().__init__()
        # --- Do all blocks
        self.block_l = []        
        for config_block in config.model.block_l:
            # --- Do all sequential
            sequential_l = []
            for config_sequential in config_block.sequential_l:
                # --- Do one sequential
                layer_l = []
                for config_layer in config_sequential.layer_l:
                    module, current_input_dim = f_parse_component(config_layer[0], config_layer[1], current_input_dim) 
                    layer_l.append( module )
                # Add to the list of sequentiak
                sequential_l.append( nn.Sequential (*layer_l) )
            # --- Add to the list of blocks
            self.block_l.append( nn.ModuleList(sequential_l) )
        self.model = nn.ModuleList(self.block_l)

    def forward(self, X, do_verbose=False):
        store_d = {}
        for idx_block, block in enumerate(self.model):
            for idx_sequential, sequential in enumerate(block):
                if do_verbose:   print(f'{idx_block}/{idx_sequential}---------------------------------\n{sequential}\n> in: {X.size()}')
                if isinstance(sequential[0], nnStoreAs): store_d[sequential[0].key] = X
                elif isinstance(sequential[0], nnCatWith): X = torch.cat( ( X, store_d[sequential[0].key]), dim=1)
                else: X = sequential( X )
                if do_verbose: print(f'> out: {X.size()}')
        return X