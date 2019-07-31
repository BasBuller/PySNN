import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from snn.utils import _set_no_grad
import snn.functional as sf


#########################################################
# Linear layer
#########################################################
class Connection(nn.Module):
    r"""Base class for defining SNN connection/layer.

    This object connects layers of neurons, it also contains the synaptic weights.
    """
    def __init__(self,
                 in_features,
                 out_features,
                 alpha_t,
                 tau_t,
                 dt,
                 delay):
        super(Connection, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # TODO: Make it such that if delay is 0 no delay Tensor is made.
        if isinstance(delay, int):
            if delay == 0:
                delay_init = None
            else:
                delay_init = torch.ones(out_features, in_features) * delay
        elif isinstance(delay, torch.FloatTensor):
            delay_init = torch.tensor(delay, dtype=torch.float)
        else:
            raise TypeError("Incorrect data type provided for delay_init, please provide an int or FloatTensor")

        # Fixed parameters
        self.dt = Parameter(torch.tensor(dt, dtype=torch.float))
        self.tau_t = Parameter(torch.tensor(tau_t, dtype=torch.float))
        self.alpha_t = Parameter(torch.tensor(alpha_t, dtype=torch.float))

        # Learnable parameters
        self.delay_init = Parameter(delay_init) if delay_init is not None else None

        # State parameters
        self.trace = Parameter(torch.Tensor(out_features, in_features))
        self.delay = Parameter(torch.Tensor(out_features, in_features))

    def convert_spikes(self, x):
        r"""Convert input from Byte Tensor to same data type as the weights."""
        return x.type(self.weight.dtype)

    def propagate_spike(self, x):
        r"""Track propagation of spikes through synapses if the connection."""
        self.delay[self.delay > 0] -= 1
        spike_out = self.delay == 1
        self.delay += torch.matmul(self.delay_init, x)
        return self.convert_spikes(spike_out)

    def no_grad(self):
        r"""Set require_gradients to False and turn off training mode."""
        _set_no_grad(self)
        self.train(False)

    def reset_state(self):
        r"""Set state Parameters (e.g. trace) to their resting state."""
        self.trace.data.fill_(0)
        self.delay.data.fill_(0)

    def reset_parameters(self):
        r"""Reinnitialize learnable network Parameters (e.g. weights)."""
        self.delay_init.data.fill_(0)

    def init_connection(self):
        r"""Collection of all intialization methods."""
        assert hasattr(self, "weight"), "Weight attribute is missing for {}.".format(
            self.__class__.__name__)
        assert isinstance(self.weight, Parameter), "Weight attribute is not a PyTorch Parameter for {}.".format(
            self.__class__.__name__)
        self.reset_state()
        self.reset_parameters()
        self.no_grad()


#########################################################
# Linear layer
#########################################################
class LinearDelayed(Connection):
    r"""SNN linear (fully connected) layer with same interface as torch.nn.Linear."""
    def __init__(self, in_features, out_features, alpha_t, tau_t, dt, delay):
        super(LinearDelayed, self).__init__(in_features, out_features, dt, delay, tau_t, alpha_t)

        # Define parameters
        self.weight = Parameter(torch.Tensor(out_features, in_features))

        # Initialize connection
        self.init_connection()

    def reset_parameters(self):
        r"""Reinnitialize network Parameters (e.g. weights)."""
        super().reset_parameters()
        nn.init.uniform_(self.weight)

    def update_trace(self, x):
        self.trace.data = sf._exponential_trace_update(self.trace, x, self.alpha_t, self.tau_t,
            self.dt)

    def forward(self, x):
        x = self.convert_spikes(x)
        x = self.propagate_spike(x)
        return (self.weight * x), self.trace.data
