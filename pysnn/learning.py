import torch
import torch.nn as nn

from pysnn.connection import Connection
from pysnn.utils import _set_no_grad, tensor_clamp


#########################################################
# Learning rule base class
#########################################################
class LearningRule(nn.Module):
    r"""Base class for correlation based learning rules in spiking neural networks."""

    def __init__(self, connection, lr):
        super(LearningRule, self).__init__()
        self.connection = connection
        self.register_buffer("lr", torch.tensor(lr, dtype=torch.float))

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
        self.register_buffer("w_init", torch.tensor(w_init, dtype=torch.float))
        self.register_buffer("a", torch.tensor(a, dtype=torch.float))

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
    r"""Apply MSTDPET from (Florian 2007) to the provided connections.
    
    Uses just a single, scalar reward value.
    Update rule can be applied at any desired time step.
    """

    def __init__(
        self, connection, pre_neuron, post_neuron, a_pre, a_post, lr, e_trace_decay
    ):
        super(MSTDPET, self).__init__(connection, lr)
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron

        self.register_buffer(
            "e_trace_decay", torch.tensor(e_trace_decay, dtype=torch.float)
        )
        self.register_buffer("a_pre", torch.tensor(a_pre, dtype=torch.float))
        self.register_buffer("a_post", torch.tensor(a_post, dtype=torch.float))

        self.register_buffer("e_trace", torch.Tensor(*self.connection.weight.shape))

        self.init_rule()

    def reset_state(self):
        self.e_trace.fill_(0)

    def update_eligibility_trace(self):
        r"""Update eligibility trace based on pre and postsynaptic spiking activity.
        
        This function has to be called manually after each timestep. Should not be called from within forward, 
        as this does is likely not called every timestep.
        """

        self.e_trace *= self.e_trace_decay
        self.e_trace += self.a_pre * torch.ger(
            self.post_neuron.spikes.view(-1).float(), self.pre_neuron.trace.view(-1)
        ) - self.a_post * torch.ger(
            self.post_neuron.trace.view(-1), self.pre_neuron.spikes.view(-1).float()
        )

    def forward(self, reward):
        # TODO: add weight clamping?
        self.connection.weight += self.lr * reward * self.e_trace
