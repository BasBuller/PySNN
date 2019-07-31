import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair

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
                 shape,
                 dt,
                 delay):
        super(Connection, self).__init__()
        self.shape = shape

        if isinstance(delay, int):
            if delay == 0:
                delay_init = None
            else:
                delay_init = torch.ones(*shape) * delay
        elif isinstance(delay, torch.Tensor):
            delay_init = delay
        else:
            raise TypeError("Incorrect data type provided for delay_init, please provide an int or FloatTensor")

        # Fixed parameters
        self.dt = Parameter(torch.tensor(dt, dtype=torch.float))

        # Learnable parameters
        if delay_init is not None:
            self.delay_init = Parameter(delay_init)
        else:
            self.register_parameter("delay_init", None)

        # State parameters
        self.trace = Parameter(torch.Tensor(*shape))
        self.delay = Parameter(torch.Tensor(*shape))

    def convert_spikes(self, x):
        r"""Convert input from Byte Tensor to same data type as the weights."""
        return x.type(self.weight.dtype)

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
class LinearExponentialDelayed(Connection):
    r"""SNN linear (fully connected) layer with interface comparable to torch.nn.Linear."""
    def __init__(self,
                 in_features,
                 out_features,
                 dt,
                 delay,
                 tau_t,
                 alpha_t):
        self.shape = (out_features, in_features)
        super(LinearExponentialDelayed, self).__init__(self.shape, dt, delay)

        # Fixed parameters
        self.tau_t = Parameter(torch.tensor(tau_t, dtype=torch.float))
        self.alpha_t = Parameter(torch.tensor(alpha_t, dtype=torch.float))

        # Learnable parameters
        self.weight = Parameter(torch.Tensor(*self.shape))

        # Initialize connection
        self.init_connection()

    def reset_parameters(self):
        r"""Reinnitialize network Parameters (e.g. weights)."""
        super().reset_parameters()
        nn.init.uniform_(self.weight)

    def update_trace(self, x):
        r"""Update trace according to exponential decay function and incoming spikes."""
        self.trace.data = sf._exponential_trace_update(self.trace, x.t(), self.alpha_t, self.tau_t,
            self.dt)

    def propagate_spike(self, x):
        r"""Track propagation of spikes through synapses if the connection."""
        self.delay[self.delay > 0] -= 1
        spike_out = self.delay == 1
        self.delay += torch.matmul(self.delay_init, x)
        return self.convert_spikes(spike_out)

    def forward(self, x):
        x = self.convert_spikes(x)
        self.update_trace(x)  # TODO: Check if this is correct position
        x = self.propagate_spike(x)
        return (self.weight * x), self.trace.data


#########################################################
# Convolutional layer
#########################################################
class Conv2dExponentialDelayed(Connection):
    r"""Convolutional SNN layer interface comparable to torch.nn.Conv2d."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dt,
                 delay,
                 tau_t,
                 alpha_t,
                 stride=1,
                 padding=0,
                 dilation=1):
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.shape = (out_channels, in_channels, *self.kernel_size)
        super(Conv2dExponentialDelayed, self).__init__(self.shape, dt, delay)

        # Fixed parameters
        self.tau_t = Parameter(torch.tensor(tau_t, dtype=torch.float))
        self.alpha_t = Parameter(torch.tensor(alpha_t, dtype=torch.float))

        # Weight parameter
        self.weight = Parameter(torch.Tensor(*self.shape))
        self.register_parameter("bias", None)

        # Intialize layer
        self.init_connection()

    def reset_parameters(self):
        r"""Reinnitialize network Parameters (e.g. weights)."""
        super().reset_parameters()
        nn.init.uniform_(self.weight)

    def update_trace(self, x):
        r"""Update trace according to exponential decay function and incoming spikes."""
        # TODO: Have to permute x before function call, currently dimensions ordered incorrectly
        self.trace.data = sf._exponential_trace_update(self.trace, x, self.alpha_t, self.tau_t,
            self.dt)

    def propagate_spike(self, x):
        r"""Track propagation of spikes through synapses if the connection."""
        self.delay[self.delay > 0] -= 1
        spike_out = self.delay == 1
        self.delay += F.conv2d(x, self.delay_init, self.bias, self.stride, self.padding, self.dilation)
        return self.convert_spikes(spike_out)

    def forward(self, x):
        x = self.convert_spikes(x)
        self.update_trace(x)
        x = self.propagate_spike(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)
        return x, self.trace.data
