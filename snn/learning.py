import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from snn.utils import _set_no_grad


#########################################################
# Fede STDP
#########################################################
class FedeSTDP(nn.Module):
    def __init__(self,
                 connections,
                 lr,
                 w_init,
                 a):
        super(FedeSTDP, self).__init__()
        self.connections = connections
        self.w_init = Parameter(torch.tensor(w_init, dtype=torch.float))
        self.lr = Parameter(torch.tensor(lr, dtype=torch.float))
        self.a = Parameter(torch.tensor(a, dtype=torch.float))

        # TODO: within __init__ determine if dealing with conv or linear 
        # and determine corresponding trace transform. Especially conv needs
        # some funky transform before it can be summed.

        self.no_grad()

    def no_grad(self):
        _set_no_grad(self)
        self.train(False)

    def reset_state(self):
        pass

    def forward(self):
        for connection in self.connections:
            w = connection.weight.data
            trace = connection.trace.data
            trace = trace.view(-1, *w.shape)

            # LTP computation
            ltp_w = torch.exp(-(w - self.w_init))
            ltp_t = torch.exp(trace) - self.a
            ltp = ltp_w * ltp_t

            # LTD computation
            ltd_w = -torch.exp(w - self.w_init)
            ltd_t = torch.exp(1 - trace) - self.a
            ltd = ltd_w * ltd_t

            # Perform weight update
            connection.weight.data += self.lr * (ltp + ltd).mean(0)
