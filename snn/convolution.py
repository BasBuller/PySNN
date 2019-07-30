import torch
import torch.nn as nn
import torch.nn.functional as torch_f


class Conv(nn.Module):
    def __init__(self):
        return

    def forward(self, input):
        return torch_f.conv3d(input)