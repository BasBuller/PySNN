import torch
import torch.nn as nn

from pysnn.connection import Connection
from pysnn.utils import _set_no_grad, tensor_clamp


#########################################################
# Learning rule base class
#########################################################
class LearningRule(nn.Module):
    r"""Base class for correlation based learning rules in spiking neural networks.
    
    Each 'layer' (combination of a Connection and a Neuron object) needs a separate LearningRule object.
    """
    # TODO: use Black (install it) and see whether this changes back to correct
    def __init__(self,
                 connection,
                 lr):
        super(LearningRule, self).__init__()
        self.connection = connection
        self.lr = lr

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
class FedeSTDP(LearningRule):
    r"""STDP version for Paredes Valles, performs mean operation over the batch 
    dimension before weight update."""
    def __init__(self, connection, lr, w_init, a):
        super(FedeSTDP, self).__init__(connection, lr)
        self.w_init = torch.tensor(w_init, dtype=torch.float)
        self.a = torch.tensor(a, dtype=torch.float)

        self.init_rule()

    def forward(self):
        w = self.connection.weight.data
        trace = self.connection.trace.data.view(-1, *w.shape)

        # LTP and LTD
        dw = w - self.w_init

        # LTP computation
        ltp_w = torch.exp(-dw)
        ltp_t = torch.exp(trace) - self.a
        ltp = ltp_w * ltp_t

        # LTD computation
        ltd_w = -torch.exp(dw)
        ltd_t = torch.exp(1 - trace) - self.a
        ltd = ltd_w * ltd_t

        # Perform weight update
        self.connection.weight += self.lr * (ltp + ltd).mean(0)


#########################################################
# MSTDPET
#########################################################
class MSTDPET(LearningRule):
    r"""Apply MSTDPET from (Florian 2007) to the provided connections."""
    def __init__(self, connection, pre_neuron, post_neuron, lr, dt, e_trace_decay):
        super(MSTDPET, self).__init__(connection, post_neuron, lr)
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron

        self.dt = dt
        self.e_trace_decay = e_trace_decay

        self.e_trace = torch.Tensor(*self.connection.weight.shape)

        self.init_rule()

    def reset_state(self):
        self.e_trace.fill_(0)

    def update_eligibility_trace(self):
        self.e_trace *= self.e_trace_decay
        # TODO: check dimensions
        # TODO: does spiking() still give spikes or have these been reset?
        self.e_trace += self.pre_neuron.trace * self.post_neuron.spiking() - self.post_neuron.trace * self.pre_neuron.spiking()

    def forward(self, reward):
        # TODO: add weight clamping?
        self.connection.weight += self.lr * reward * self.e_trace
