import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from snn.connection import Connection
from snn.utils import _set_no_grad


#########################################################
# Learning rule base class
#########################################################
class LearningRule(nn.Module):
    r"""Base class for correlation based learning rules in spiking neural networks."""
    def __init__(self,
                 connections,
                 lr):
        super(LearningRule, self).__init__()
        self.connections = connections
        self.lr = Parameter(torch.tensor(lr, dtype=torch.float))

    def no_grad(self):
        _set_no_grad(self)

    def reset_state(self):
        pass

    def init_rule(self):
        self.no_grad()
        self.reset_state()


#########################################################
# Fede STDP
#########################################################
def fede_ltp_ltd(w, w_init, trace, a):
    # LTP computation
    ltp_w = torch.exp(-(w - w_init))
    ltp_t = torch.exp(trace) - a
    ltp = ltp_w * ltp_t

    # LTD computation
    ltd_w = -torch.exp(w - w_init)
    ltd_t = torch.exp(1 - trace) - a
    ltd = ltd_w * ltd_t

    return ltp, ltd


class FedeSTDP(LearningRule):
    r"""STDP version for Paredes Valles, performs mean operation over the batch dimension before weight update."""
    def __init__(self,
                 connections,
                 lr,
                 w_init,
                 a):
        # Make sure conncetions is an iterable for compatibility with forward function
        if isinstance(connections, Connection):
            connections = (connections)
        super(FedeSTDP, self).__init__(connections, lr)
        self.w_init = Parameter(torch.tensor(w_init, dtype=torch.float))
        self.a = Parameter(torch.tensor(a, dtype=torch.float))

        self.init_rule()

    def forward(self):
        for connection in self.connections:
            w = connection.weight.data
            trace = connection.trace.data.view(-1, *w.shape)

            # LTP and LTD
            ltp, ltd = fede_ltp_ltd(w, self.w_init, trace, self.a)

            # Perform weight update
            connection.weight.data += self.lr * (ltp + ltd).mean(0)


#########################################################
# RSTDP for output class only (Mozafari et al.)
#########################################################
class RSTDPRateSingleElement(LearningRule):
    def __init__(self,
                 connection,
                 lr,
                 w_init,
                 a,
                 out_shape,
                 start_counter):
        super(RSTDPRateSingleElement, self).__init__(connection, lr)
        self.w_init = Parameter(torch.tensor(w_init, dtype=torch.float))
        self.a = Parameter(torch.tensor(a, dtype=torch.float))
        self.activity = Parameter(torch.zeros(*out_shape))
        self.counter = 0
        self.start_counter = start_counter

        self.init_rule()

    def reset_state(self):
        self.activity.fill_(0)
        self.counter = 0

    def forward(self, x, label):
        if self.counter >= self.start_counter:
            w = self.connections.weight.data
            trace = self.connections.trace.data.view(-1, *w.shape)

            # Determine if output class is correct
            reward = torch.ones_like(label)
            _, m_ind = x.max(-1)
            reward[m_ind != label] = -1

            # LTP and LTD
            ltp, ltd = fede_ltp_ltd(w, self.w_init, trace, self.a)

            # Swap LTP and LTD for faulty network output
            store = ltp[reward==-1]
            ltp[reward==-1] = ltd[reward==-1]
            ltd[reward==-1] = store

            # Update weights
            delta_w = self.lr * (ltp + ltd)
            mask = torch.zeros_like(delta_w)
            mask[label, :] = 1
            delta_w *= mask
            self.connections.weight.data += delta_w.mean(0)

        self.activity += x.to(self.activity.dtype)
        self.counter += 1
