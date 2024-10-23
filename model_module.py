#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: model_module.py
Description: modules (SincNet, ConvNext, Depth-Wsie Separable, TCN) to be used by model_factory
Author: geoffroy.peeters@telecom-paris.fr
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm

# ConvNeXt PAPER: https://arxiv.org/pdf/2201.03545
# ConvNeXt CODE: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, drop_path=0.0):
        super(ConvNeXtBlock, self).__init__()
        
        # 1. Depthwise convolution (spatial convolution with large kernel)
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=in_channels)
        
        # 2. Layer normalization applied across channels
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)  # LayerNorm is applied after permuting to (B, C, H, W)
        
        # 3. Pointwise convolution to project to higher dimensions (expanding and compressing channels)
        self.pwconv1 = nn.Linear(in_channels, 4 * in_channels)  # expand channels by 4x
        self.act = nn.GELU()  # GELU activation
        self.pwconv2 = nn.Linear(4 * in_channels, out_channels)  # project back to original channels
        
        # 4. Stochastic depth (optional) for better regularization
        self.drop_path = nn.Identity() if drop_path == 0 else StochasticDepth(drop_path)
    
    def forward(self, x):
        # Input: (B, C, H, W)
        residual = x

        # 1. Depthwise convolution
        x = self.dwconv(x)
        
        # 2. LayerNorm after permute to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        
        # 3. Pointwise convolutions + GELU
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # 4. Drop path (if applicable) and residual connection
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        x = self.drop_path(x) + residual  # Add residual connection
        
        return x

class StochasticDepth(nn.Module):
    """Drop paths (stochastic depth) per sample (when applied in the main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        # Sample binary mask
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)
        return x / keep_prob * binary_mask




# TCN paper: https://arxiv.org/pdf/1803.01271
# TCN code: https://github.com/locuslab/TCN
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)




# Paper: 
# Code: 
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = 'same'),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = 1, padding = 'same'),
                        nn.BatchNorm2d(out_channels))
        self.downsample = False
        
        if in_channels != out_channels:
            self.downsample = True
            self.conv_ds = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 'same')
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample: residual = self.conv_ds(x)
        out += residual
        out = self.relu(out)
        return out



# Paper: 
# Code: https://github.com/seungjunlee96/Depthwise-Separable-Convolution_Pytorch/blob/master/DepthwiseSeparableConvolution/DepthwiseSeparableConvolution.py
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout, kernel_size=3, padding=1):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out





# Code: https://github.com/mravanelli/SincNet/tree/master to perform SincNet front-end
class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv_fast,self).__init__()

        if in_channels != 1:
            #msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0: self.kernel_size=self.kernel_size+1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias: raise ValueError('SincConv does not support bias.')
        if groups > 1: raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)
        mel_v = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1)
        hz_v = self.to_hz(mel_v)
        

        # filter lower frequency (out_channels, 1)
        self.low_hz_v_ = nn.Parameter(torch.Tensor(hz_v[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_v_ = nn.Parameter(torch.Tensor(np.diff(hz_v)).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_ = 0.54-0.46*torch.cos(2*np.pi*n_lin/self.kernel_size);


        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*np.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes

 


    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        low_v = self.min_low_hz  + torch.abs(self.low_hz_v_)
        high_v = torch.clamp(low_v + self.min_band_hz + torch.abs(self.band_hz_v_), 
                           self.min_low_hz,
                           self.sample_rate/2)
        band_v = (high_v - low_v)[:,0]
        
        f_times_t_low = torch.matmul(low_v, self.n_)
        f_times_t_high = torch.matmul(high_v, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_/2)) * self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations. 
        band_pass_center = 2 * band_v.view(-1,1)
        band_pass_right= torch.flip(band_pass_left, dims=[1])
        
        band_pass=torch.cat([band_pass_left,
                             band_pass_center,
                             band_pass_right],dim=1)

        band_pass = band_pass / (2*band_v[:,None])

        self.filters = (band_pass).view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride, padding=self.padding, dilation=self.dilation, bias=None, groups=1) 