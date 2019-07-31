from itertools import repeat

import torch
import torch.nn as nn


#########################################################
# Class initialization
#########################################################
def _set_no_grad(module):
    for param in module.parameters():
        param.requires_grad = False

def _reset_state(module):
    for param in module.parameters():
        nn.init.uniform_(param, 0, 0)
