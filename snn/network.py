from functools import   wraps

import torch
import torch.nn as nn
import torch.nn.functional as F


#########################################################
# Event stream decorator
#########################################################
def snn_forward(forward):
    @wraps(forward)
    def wrapper(self, input):
        out = []
        for x in input:
            out.append(forward(x))
        return torch.stack(out, dim=0)
    return wrapper 