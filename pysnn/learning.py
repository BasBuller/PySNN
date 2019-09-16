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
    def __init__(self,
                 connections,
                 lr):
        super(LearningRule, self).__init__()
        self.connections = connections
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
    def __init__(self,
                 connections,
                 lr,
                 w_init,
                 a):
        # Make sure connections is an iterable for compatibility with forward function
        if isinstance(connections, Connection):
            connections = (connections)
        super(FedeSTDP, self).__init__(connections, lr)
        self.w_init = torch.tensor(w_init, dtype=torch.float)
        self.a = torch.tensor(a, dtype=torch.float)

        self.init_rule()

    def forward(self):
        for connection in self.connections:
            w = connection.weight.data
            trace = connection.trace.data.view(-1, *w.shape)

            # LTP and LTD
            ltp, ltd = _fede_ltp_ltd(w, self.w_init, trace, self.a)

            # Perform weight update
            connection.weight.data += self.lr * (ltp + ltd).mean(0)


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
    def __init__(self, 
                 connections, 
                 lr,
                 dt):
        super(MSTDPET, self).__init__(connections, lr)
        self.dt = dt

    def forward(self, reward):
        for conn in self.connections:
            loc_reward = reward[conn["reward_type"]]

            trace = conn["pre_syn_trace"] - conn["post_syn_trace"].permute(0, 2, 1)
            delta_w = self.lr * self.dt * loc_reward * trace

            conn["weights"] += delta_w.mean(0)
            conn["weights"].data = tensor_clamp(conn["weights"], conn["w_min"], conn["w_max"])
            

#########################################################
# Additive RSTDP
#########################################################
class AdditiveRSTDPLinear(LearningRule):
    r"""Basic, additive RSTDP formulation for linear layers.
    
    Can be used for many published learning rules, the difference lies in the trace formulations.
    Use a neuron with the desired trace formulation in conjunction with this learning rule.
    """
    def __init__(self,
                 connections,
                 lr,
                 dt):
        super(AdditiveRSTDPLinear, self).__init__(connections, lr)
        self.dt = dt

    def forward(self, reward):
        for conn in self.connections:
            trace = conn["pre_syn_trace"] - conn["post_syn_trace"].permute(0, 2, 1)
            delta_w = self.lr * self.dt * reward * trace

            conn["weights"] += delta_w.mean(0)
            # conn["weights"].clamp_(conn["w_min"], conn["w_max"])
            conn["weights"] = tensor_clamp(conn["weights"], conn["w_min"], conn["w_max"])
