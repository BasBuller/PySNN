import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from snn.utils import _set_no_grad


#########################################################
# Linear layer
#########################################################
class Connection(nn.Module):
    r"""Base class for defining SNN connection/layer.
    
    This object connects layers of neurons, it also contains the synaptic weights.
    """
    def __init__(self, in_features, out_features, dt, delay):
        super(Connection, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dt = dt

        if isinstance(delay, int):
            self.delay_init = torch.ones(out_features, in_features) * delay
        elif isinstance(delay, torch.FloatTensor):
            self.delay_init = delay
        else:
            raise TypeError("Incorrect data type provided for delay_init, please provide an int or FloatTensor")

        # Parameters
        self.trace = Parameter(torch.Tensor(out_features, in_features))
        self.delay = Parameter(torch.Tensor(out_features, in_features))  # TODO: Look into using uint8 here

        self.reset_state()

    def convert_input(self, x):
        """Convert input from Byte Tensor to same data type as the weights."""
        return x.type(self.weight.dtype)

    def propagate_spike(self, x):
        """Track propagation of spikes through synapses if the connection."""
        self.delay[self.delay > 0] -= 1
        spike_out = self.delay == 1
        x = x.squeeze(0)
        self.delay[:, x] = self.delay_init[:, x]
        return spike_out.float()

    def no_grad(self):
        """Set require_gradients to False and turn off training mode."""
        _set_no_grad(self)
        self.train(False)

    def reset_state(self):
        """Set state Parameters (e.g. trace) to their resting state."""
        self.trace.data.fill_(0)

    def reset_parameters(self):
        """Reinnitialize network Parameters (e.g. weights)."""
        nn.init.uniform_(self.weight)
        self.delay.data.fill_(0)


#########################################################
# Linear layer
#########################################################
class Linear(Connection):
    r"""SNN linear (fully connected) layer with same interface as torch.nn.Linear."""
    def __init__(self, in_features, out_features, dt, delay):
        super(Linear, self).__init__(in_features, out_features, dt, delay)

        # Define parameters
        self.weight = Parameter(torch.Tensor(out_features, in_features))

        # Turn off gradients and training mode
        self.reset_parameters()
        self.no_grad()

    def forward(self, x):
        x = self.propagate_spike(x)
        return (self.weight * x).sum(1).unsqueeze(0)
