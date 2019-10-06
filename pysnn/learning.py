import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from pysnn.connection import Connection
from pysnn.utils import _set_no_grad, tensor_clamp


#########################################################
# Learning rule base class
#########################################################
class LearningRule(nn.Module):
    r"""Base class for correlation based learning rules in spiking neural networks."""

    def __init__(self, connection, pre_neuron, post_neuron, lr):
        super(LearningRule, self).__init__()
        self.connection = connection
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
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
def _fede_ltp_ltd(w, w_init, trace, a):
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
    r"""STDP version for Paredes Valles, performs mean operation over the batch 
    dimension before weight update."""

    def __init__(self, connection, pre_neuron, post_neuron, lr, w_init, a):
        super(FedeSTDP, self).__init__(connection, pre_neuron, post_neuron, lr)
        self.w_init = torch.tensor(w_init, dtype=torch.float)
        self.a = torch.tensor(a, dtype=torch.float)

        self.init_rule()

    def forward(self):
        w = self.connection.weight.data
        trace = self.connection.trace.data.view(-1, *w.shape)

        # LTP and LTD
        ltp, ltd = _fede_ltp_ltd(w, self.w_init, trace, self.a)

        # Perform weight update
        self.connection.weight.data += self.lr * (ltp + ltd).mean(0)


#########################################################
# MSTDPET
#########################################################
class MSTDPET(LearningRule):
    r"""Apply MSTDPET from (Florian 2007) to the provided connections.
    
    Uses just a single, scalar reward value.
    Update rule can be applied at any desired time step.

    The connection parameter is a list of dictionaries, each dict containing the following keys:
        - weights: weight Parameter for connection.
        - pre_syn_trace: pre synaptic traces.
        - post_syn_trace: post synaptic traces.
    """

    def __init__(
        self, connection, pre_neuron, post_neuron, a_pre, a_post, lr, dt, e_trace_decay
    ):
        super(MSTDPET, self).__init__(connection, pre_neuron, post_neuron, lr)
        self.dt = dt
        self.e_trace_decay = e_trace_decay
        self.a_pre = a_pre
        self.a_post = a_post

        self.e_trace = torch.Tensor(*self.connection.weight.shape)

        self.init_rule()

    def reset_state(self):
        self.e_trace.fill_(0)

    def update_eligibility_trace(self):
        self.e_trace *= self.e_trace_decay
        # TODO: check dimensions
        # TODO: does spiking() still give spikes or have these been reset?
        self.e_trace += self.a_pre * torch.ger(
            self.post_neuron.spikes.view(-1).float(), self.pre_neuron.trace.view(-1)
        ) - self.a_post * torch.ger(
            self.post_neuron.trace.view(-1), self.pre_neuron.spikes.view(-1).float()
        )

    def forward(self, reward):
        # TODO: why would one use .data? Also done in FedeSTDP
        # TODO: add weight clamping?
        # TODO: not sure whether dt belongs here
        self.update_eligibility_trace()
        self.connection.weight.data += self.lr * self.dt * reward * self.e_trace


#########################################################
# Additive RSTDP
#########################################################
class AdditiveRSTDPLinear(LearningRule):
    r"""Basic, additive RSTDP formulation for linear layers.
    
    Can be used for many published learning rules, the difference lies in the trace formulations.
    Use a neuron with the desired trace formulation in conjunction with this learning rule.
    """

    def __init__(self, connection, pre_neuron, post_neuron, lr, dt):
        super(AdditiveRSTDPLinear, self).__init__(
            connection, pre_neuron, post_neuron, lr
        )
        self.dt = dt

    def forward(self, reward):
        # TODO: needs transpose of second
        # TODO: pre - post or other way around?
        delta_trace = self.pre_neuron.trace - self.post_neuron.trace
        # TODO: not sure whether dt belongs here
        # TODO: needs weight clamp
        self.connection.weight += self.lr * self.dt * reward * delta_trace
