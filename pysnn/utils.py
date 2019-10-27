import torch

from torch.nn.modules.utils import _pair


#########################################################
# Class initialization
#########################################################
def _set_no_grad(module):
    for param in module.parameters():
        param.requires_grad = False


#########################################################
# Convolutional shapes
#########################################################
def conv2d_output_shape(h_in, w_in, kernel_size, stride=1, padding=0, dilation=1):
    kernel_size = _pair(kernel_size)
    padding = _pair(padding)
    dilation = _pair(dilation)
    stride = _pair(stride)
    h_out = (
        (h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]
    ) + 1
    w_out = (
        (w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]
    ) + 1
    return (int(h_out), int(w_out))


#########################################################
# Interspike time
#########################################################
def interspike_time(spike_array):
    spike_indices = (spike_array > 0).nonzero()
    interspike_times = []
    for idx in range(len(spike_indices) - 1):
        inter_time = spike_indices[idx + 1] - spike_indices[idx]
        interspike_times.append(inter_time)
    return interspike_times


#########################################################
# Tensor clamping
#########################################################
def tensor_clamp(tensor, min, max):
    clamp = torch.min(tensor, max)  # upper boundary
    clamp = torch.max(clamp, min)  # lower boundary
    return clamp
