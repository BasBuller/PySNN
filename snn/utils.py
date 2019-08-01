from itertools import repeat

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair


#########################################################
# Class initialization
#########################################################
def _set_no_grad(module):
    for param in module.parameters():
        param.requires_grad = False

def _reset_state(module):
    for param in module.parameters():
        nn.init.uniform_(param, 0, 0)


#########################################################
# Convolutional shapes
#########################################################
def conv2d_output_shape(h_in, w_in, kernel_size, stride=1, padding=0, dilation=0):
    kernel_size = _pair(kernel_size)
    padding = _pair(padding)
    dilation = _pair(dilation)
    stride = _pair(stride)
    h_out = ((h_in + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1) / stride[0]) + 1
    w_out = ((w_in + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1) / stride[1]) + 1
    return (int(h_out), int(w_out))
