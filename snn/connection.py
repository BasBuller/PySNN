import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair

from snn.utils import _set_no_grad, conv2d_output_shape
import snn.functional as sF


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
        self.synapse_shape = shape

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
        # self.train(False)

    def reset_state(self):
        r"""Set state Parameters (e.g. trace) to their resting state."""
        self.trace.fill_(0)
        self.delay.fill_(0)

    def reset_parameters(self):
        r"""Reinnitialize learnable network Parameters (e.g. weights)."""
        if self.delay_init is not None:
            self.delay_init.fill_(0)
        nn.init.uniform_(self.weight)

    def init_connection(self):
        r"""Collection of all intialization methods.
        
        Assumes weights are implemented by the class that inherits from this base class.
        """
        assert hasattr(self, "weight"), "Weight attribute is missing for {}.".format(
            self.__class__.__name__)
        assert isinstance(self.weight, Parameter), "Weight attribute is not a PyTorch Parameter for {}.".format(
            self.__class__.__name__)
        self.no_grad()
        self.reset_state()
        self.reset_parameters()

    def propagate_spike(self, x):
        r"""Track propagation of spikes through synapses if the connection."""
        if self.delay_init is not None:
            x = sF._spike_delay_update(self.delay, self.delay_init, x)
        return self.convert_spikes(x)


#########################################################
# Linear layer
#########################################################
class LinearExponentialDelayed(Connection):
    r"""SNN linear (fully connected) layer with interface comparable to torch.nn.Linear."""
    def __init__(self,
                 in_features,
                 out_features,
                 batch_size,
                 dt,
                 delay,
                 tau_t,
                 alpha_t):
        # Assertions
        assert delay != 0, "Delay is zero, please use LinearExponential instead."

        # Dimensions
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size

        self.synapse_shape = (batch_size, out_features, in_features)
        super(LinearExponentialDelayed, self).__init__(self.synapse_shape, dt, delay)

        # Fixed parameters
        self.tau_t = Parameter(torch.tensor(tau_t, dtype=torch.float))
        self.alpha_t = Parameter(torch.tensor(alpha_t, dtype=torch.float))

        # Learnable parameters
        self.weight = Parameter(torch.Tensor(out_features, in_features))

        # Initialize connection
        self.init_connection()

    # def reset_parameters(self):
    #     r"""Reinnitialize network Parameters (e.g. weights)."""
    #     super().reset_parameters()
    #     nn.init.uniform_(self.weight)

    # Support function
    def fold_traces(self):
        return self.trace.data.view(self.batch_size, -1, self.out_features)

    # Standard functions
    def update_trace(self, x):
        r"""Update trace according to exponential decay function and incoming spikes."""
        sF._connection_exponential_trace_update(self.trace, x, self.alpha_t, self.tau_t,
                                     self.dt)

    def activation_potential(self, x):
        r"""Determine activation potentials from each synapse for current time step."""
        out = (x * self.weight).sum(2, keepdim=True)
        return out.view(self.batch_size, -1, self.out_features)

    def forward(self, x):
        x = self.convert_spikes(x)
        self.update_trace(x)
        x = self.propagate_spike(x)
        return self.activation_potential(x), self.fold_traces()


#########################################################
# Convolutional layer
#########################################################
class Conv2dExponentialDelayed(Connection):
    r"""Convolutional SNN layer interface comparable to torch.nn.Conv2d."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 h_in,
                 w_in,
                 batch_size,
                 dt,
                 delay,
                 tau_t,
                 alpha_t,
                 stride=1,
                 padding=0,
                 dilation=1):
        # Assertions
        assert delay != 0, "Delay is zero, please use Conv2dExponential instead."

        # Image paramters
        # self.in_channels = in_channels
        # self.h_in = h_in
        # self.w_in = w_in
        # self.image_in_size = (h_in, w_in)

        # Convolution parameters
        self.batch_size = batch_size
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        # Synapse connections shape
        empty_input = torch.zeros(batch_size, in_channels, h_in, w_in)
        synapse_shape = list(self.unfold(empty_input).shape)
        synapse_shape[1] = out_channels
        self.synapse_shape = synapse_shape

        # Super init
        super(Conv2dExponentialDelayed, self).__init__(synapse_shape, dt, delay)

        # Output image size
        self.image_out_size = conv2d_output_shape(h_in, w_in, self.kernel_size, self.stride,
                                                   self.padding, self.dilation)

        # Fixed parameters
        self.tau_t = Parameter(torch.tensor(tau_t, dtype=torch.float))
        self.alpha_t = Parameter(torch.tensor(alpha_t, dtype=torch.float))

        # Weight parameter
        self.weight = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.register_parameter("bias", None)

        # Intialize layer
        self.init_connection()

    # def reset_parameters(self):
    #     r"""Reinnitialize network Parameters (e.g. weights)."""
    #     super().reset_parameters()
    #     nn.init.uniform_(self.weight)

    # Support functions
    def unfold(self, x):
        r"""Simply unfolds incoming image according to layer parameters."""
        return F.unfold(x, self.kernel_size, self.dilation, self.padding, self.stride).unsqueeze(1)

    def fold_im(self, x):
        r"""Simply folds incoming image according to layer parameters."""
        return x.view(-1, x.shape[1], *self.image_out_size)

    def fold_traces(self):
        r"""Simply folds incoming trace according to layer parameters."""
        return self.trace.view(-1, self.batch_size, self.out_channels, *self.image_out_size)

    # Standard functions
    def update_trace(self, x):
        r"""Update trace according to exponential decay function and incoming spikes."""
        sF._connection_exponential_trace_update(self.trace, x, self.alpha_t, self.tau_t,
            self.dt)

    def activation_potential(self, x):
        r"""Determine activation potentials from each synapse for current time step."""
        x = x * self.weight.view(self.weight.shape[0], -1).unsqueeze(2)
        x = x.sum(2, keepdim=True)
        x = x.view(-1, x.shape[1], *self.image_out_size)
        return x

    def forward(self, x):
        x = self.convert_spikes(x)
        x = self.unfold(x)  # Till here it is a rather easy set of steps
        self.update_trace(x)
        x = self.propagate_spike(x)  # Output spikes
        return self.activation_potential(x), self.fold_traces()
