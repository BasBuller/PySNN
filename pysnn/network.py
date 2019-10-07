import torch
import torch.nn as nn


#########################################################
# SNN Network
#########################################################
class SNNNetwork(nn.Module):
    r"""Simple base clase for defining SNN network, contains some convenience operators
    for e.g. clearing network state after simulating a sample.
    """

    def __init__(self):
        super(SNNNetwork, self).__init__()

    def reset_state(self):
        for module in self.modules():
            if not isinstance(module, SNNNetwork):
                module.reset_state()
