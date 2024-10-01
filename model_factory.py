import torch
import torch.nn as nn
import numpy as np
from model_module import SincConv_fast
from torch.nn.utils import weight_norm


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


def f_get_activation(value):
    """ return the corresponding activation class """
    if value=='Sigmoid': 
        value = nn.Sigmoid()
    if value=='Softmax': 
        value = nn.Softmax()
    if value=='ReLU': 
        value = nn.ReLU()
    if value=='LeakyReLU': 
        value = nn.LeakyReLU()
    if value=='PReLU':
        value = nn.PReLU()
    return value


class nnAbs(nn.Module):
    """ encapsultate torch.abs """
    def __init__(self):
        super(nnAbs, self).__init__()
    def forward(self, x):
        return torch.abs(x)

class nnMean(nn.Module):
    """ encapsultate torch.mean """
    def __init__(self, dim, keepdim):
        super(nnMean, self).__init__()
        self.dim = dim
        self.keepdim = keepdim
    def forward(self, X):
        return torch.mean(X, self.dim, self.keepdim)

class nnMax(nn.Module):
    """ encapsultate torch.mean """
    def __init__(self, dim, keepdim):
        super(nnMax, self).__init__()
        self.dim = dim
        self.keepdim = keepdim
    def forward(self, X):
        out, _ = torch.max(X, self.dim, self.keepdim)
        return out

class nnSqueeze(nn.Module):
    def __init__(self, dim):
        super(nnSqueeze, self).__init__()
        self.dim = dim
    def forward(self, X):
        out = X.squeeze(dim=self.dim)
        return out

class nnPermute(nn.Module):
    """ encapsultate .permute """
    def __init__(self, shape):
        super(nnPermute, self).__init__()
        self.shape = shape
    def forward(self, X):
        return X.permute(self.shape)

class nnBatchNorm1dT(nn.Module):
    """ perform BatchNorm1d over transposed vector (in case input format is (B,T,C) """
    def __init__(self, num_features):
        super(nnBatchNorm1dT, self).__init__()
        self.module = nn.BatchNorm1d(num_features)
    def forward(self, X):
        return self.module(X.permute(0,2,1)).permute(0,2,1)

class nnEmpty(nn.Module):
    """ encapsultate .permute """
    def __init__(self):
        super(nnEmpty, self).__init__()
    def forward(self, X):
        return X

class nnAutoPoolWeight(nn.Module):
    def __init__(self):
        super(nnAutoPoolWeight, self).__init__()
        self.autopool_param = nn.Parameter(torch.tensor(0.).float())
    def forward(self, X):
        weights = f_autopool_weights(X, self.autopool_param)
        X = torch.sum(X * weights, dim=3, keepdim=True)
        return X

class nnAutoPoolWeightSplit(nn.Module):
    def __init__(self, C):
        super(nnAutoPoolWeightSplit, self).__init__()
        self.autopool_param = nn.Parameter(torch.tensor(0.).float())
        self.C = C
    def forward(self, X):
        weights = f_autopool_weights(X[:, int(self.C/2):], self.autopool_param)
        X = torch.sum(X[:, :int(self.C/2):] * weights, dim=3, keepdim=True)
        return X

class nnSoftmaxWeight(nn.Module):
    def __init__(self, C):
        super(nnSoftmaxWeight, self).__init__()
        self.C = C
    def forward(self, X):
        weights = torch.nn.functional.softmax(X[:, int(self.C/2):], dim=3)
        X = torch.sum(X[:, :int(self.C/2)] * weights, dim=3, keepdim=True)
        return X


def f_parse_component(type, param, current_input_dim):
    """ parse type and param parameters to create NN layers, return the new dimensions of the tensor """
    # --- FC:       B, D
    # --- Conv1D:   B, C, T
    # --- Conv2D:   B, C, H=Freq, W=Time

    #print(type, param)

    if type=='LayerNorm':
        if param.normalized_shape==-1: param.normalized_shape = current_input_dim[1:] # --- B, C, T
        module = nn.LayerNorm(normalized_shape=param.normalized_shape)
    
    elif type=='BatchNorm1d':
        if param.num_features==-1: param.num_features = current_input_dim[-1] # --- (B, D) or (B, T, D)
        if 'affine' not in param.keys(): param.affine = True
        module = nn.BatchNorm1d(num_features=param.num_features, affine=param.affine)
    
    elif type=='BatchNorm1dT':
        if param.num_features==-1: param.num_features = current_input_dim[-1] # --- (B, D) or (B, T, D)
        module = nnBatchNorm1dT(num_features=param.num_features)
    
    elif type=='BatchNorm2d':
        if param.num_features==-1: param.num_features = current_input_dim[1] # --- (B, C, H, W)
        module = nn.BatchNorm2d(num_features=param.num_features)
    
    elif type=='SincNet': # --- B, C, T
        if param.in_channels==-1: param.in_channels=current_input_dim[1] 
        module = SincConv_fast(in_channels=param.in_channels, out_channels=param.out_channels, kernel_size=param.kernel_size, stride=param.stride, sample_rate=param.sr_hz)
        current_input_dim[1] = param.out_channels
        current_input_dim[2] = f_get_next_size(current_input_dim[2], param.kernel_size, param.stride)

    elif type=='Conv1d': # --- B, C, T
        if param.in_channels==-1: param.in_channels=current_input_dim[1] 
        module = nn.Conv1d(in_channels=param.in_channels, out_channels=param.out_channels, kernel_size=param.kernel_size, stride=param.stride)
        current_input_dim[1] = param.out_channels
        current_input_dim[2] = f_get_next_size(current_input_dim[2], param.kernel_size, param.stride)

    elif type=='Conv1dTCN': # --- B, C, T
        if param.in_channels==-1: param.in_channels=current_input_dim[1] 
        module = TemporalConvNet(num_inputs=param.in_channels, num_channels=param.num_channels)
        current_input_dim[1] = param.num_channels[-1]
        
    elif type=='Conv2d': # --- B, C, H=Freq, W=Time
        if param.in_channels==-1: param.in_channels=current_input_dim[1]
        if 'padding' not in param.keys(): param.padding = 'valid'
        module = nn.Conv2d(in_channels=param.in_channels, out_channels=param.out_channels, kernel_size=param.kernel_size, stride=param.stride, padding=param.padding)
        current_input_dim[1] = param.out_channels
        if param.padding != 'same':
            current_input_dim[2] = f_get_next_size(current_input_dim[2], param.kernel_size[0], param.stride[0])
            current_input_dim[3] = f_get_next_size(current_input_dim[3], param.kernel_size[1], param.stride[1])
    
    elif type=='Conv2dDS': # --- B, C, H=Freq, W=Time
        if param.in_channels==-1: param.in_channels=current_input_dim[1]
        if 'padding' not in param.keys(): param.padding = 'valid'
        module = depthwise_separable_conv(nin=param.in_channels, kernels_per_layer=1, nout=param.out_channels, kernel_size=param.kernel_size, padding=param.padding)
        current_input_dim[1] = param.out_channels
        if param.padding != 'same':
            current_input_dim[2] = f_get_next_size(current_input_dim[2], param.kernel_size[0], param.stride[0])
            current_input_dim[3] = f_get_next_size(current_input_dim[3], param.kernel_size[1], param.stride[1])
    
    elif type=='Conv2dRes': # --- B, C, H=Freq, W=Time
        if param.in_channels==-1: param.in_channels=current_input_dim[1]
        if param.padding != 'same': print(f'only work for padding=same, got {param}')
        if param.stride != 1: print(f'only work for stride=1, got {param}')
        module = ResidualBlock(in_channels=param.in_channels, out_channels=param.out_channels, kernel_size = param.kernel_size, stride = 1)
        current_input_dim[1] = param.out_channels
        
    elif type=='ConvTranspose2d':
        if param.in_channels==-1: param.in_channels=current_input_dim[1] 
        module = nn.ConvTranspose2d(in_channels=param.in_channels, out_channels=param.out_channels, kernel_size=param.kernel_size, stride=param.stride)
        current_input_dim[1] = param.out_channels
        current_input_dim[2] *= param.stride[0]
        current_input_dim[3] *= param.stride[1]

    elif type=='MaxPool1d':  # --- B, C, T
        if 'stride' not in param.keys(): param.stride=param.kernel_size
        module = nn.MaxPool1d(kernel_size=param.kernel_size, stride=param.stride)
        current_input_dim[2] = int(np.floor(current_input_dim[2]/param.kernel_size)) 

    elif type=='MaxPool2d': # --- B, C, H=Freq, W=Time
        if 'stride' not in param.keys(): param.stride=param.kernel_size
        module = nn.MaxPool2d(kernel_size=param.kernel_size, stride=param.stride)
        current_input_dim[2] = int(np.floor(current_input_dim[2]/param.kernel_size[0]))
        current_input_dim[3] = int(np.floor(current_input_dim[3]/param.kernel_size[1]))
        
    elif type=='Linear': 
        if param.in_features==-1: param.in_features = current_input_dim[-1] # --- (B, D) or (B, T, D)
        module = nn.Linear(in_features=param.in_features , out_features=param.out_features)
        current_input_dim[-1] = param.out_features

    elif type=='Activation':   
        module = f_get_activation(param)

    elif type=='Dropout':   
        module = nn.Dropout(param.p)

    elif type=='Flatten':
        module = nn.Flatten(param.start_dim)
        current_input_dim =  [current_input_dim[0], current_input_dim[1]*current_input_dim[2]]

    elif type=='Squeeze':
        module = nnSqueeze(param.dim)
        current_input_dim = [c for c in current_input_dim if c > 1]

    elif type=='Permute':
        module = nnPermute(param.shape)
        current_input_dim =  [current_input_dim[s] for s in param.shape]
    
    elif type=='Mean':
        if 'keepdim' not in param.keys(): param.keepdim = False
        module = nnMean(param.dim, param.keepdim)
        if param.keepdim==False: current_input_dim =  [current_input_dim[0], current_input_dim[1]]

    elif type=='Max':
        if 'keepdim' not in param.keys(): param.keepdim = False
        module = nnMax(param.dim, param.keepdim)
        if param.keepdim==False: current_input_dim =  [current_input_dim[0], current_input_dim[1]]

    elif type=='AutoPoolWeight':
        module = nnAutoPoolWeight()
        current_input_dim[3] = 1
    
    elif type=='AutoPoolWeightSplit':
        module = nnAutoPoolWeightSplit(current_input_dim[1])
        current_input_dim[1] = int(current_input_dim[1]/2)
        current_input_dim[3] = 1
    
    elif type=='SoftmaxWeight':
        module = nnSoftmaxWeight(current_input_dim[1])
        current_input_dim[1] = int(current_input_dim[1]/2)
        current_input_dim[3] = 1
    

    elif type=='AbsLayer':
        module = nnAbs()
    
    elif type=='DoubleChannel': # --- for U-Net
        module = nnEmpty()
        current_input_dim[1] *= 2
    
    else:
        print(f'UNKNOWN module "{type}"')

    return module, current_input_dim

