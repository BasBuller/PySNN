import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from torch.nn.modules.pooling import _MaxPoolNd, _AdaptiveMaxPoolNd

from pysnn.utils import _set_no_grad, conv2d_output_shape
import pysnn.functional as sF


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

        # Delay values are initiated with +1, reason for this is that a spike is 'generated' once the counter reaches 1
        # A counter at 0 means the cell is not in refractory or spiking state
        if isinstance(delay, int):
            if delay == 0:
                delay_init = None
            else:
                delay_init = torch.ones(*shape) * (delay + 1)
        elif isinstance(delay, torch.Tensor):
            delay_init = delay + 1
        else:
            raise TypeError("Incorrect data type provided for delay_init, please provide an int or FloatTensor")

        # Learnable parameters
        if delay_init is not None:
            self.delay_init = Parameter(delay_init)
        else:
            self.register_parameter("delay_init", None)

        # Fixed parameters
        self.dt = Parameter(torch.tensor(dt, dtype=torch.float))

        # State parameters
        self.trace = Parameter(torch.Tensor(*shape))
        self.delay = Parameter(torch.Tensor(*shape))

    def convert_spikes(self, x):
        r"""Convert input from Byte Tensor to same data type as the weights."""
        return x.type(self.weight.dtype)

    def no_grad(self):
        r"""Set require_gradients to False and turn off training mode."""
        _set_no_grad(self)

    def reset_state(self):
        r"""Set state Parameters (e.g. trace) to their resting state."""
        self.trace.fill_(0)
        self.delay.fill_(0)

    def reset_parameters(self):
        r"""Reinnitialize learnable network Parameters (e.g. weights)."""
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
            self.delay[self.delay > 0] -= 1
            spike_out = self.delay == 1
            self.delay += self.delay_init * x
        else:
            spike_out = x
        return self.convert_spikes(spike_out)


#########################################################
# Linear Layers
#########################################################
# Base class
class _Linear(Connection):
    r"""SNN linear base class, comparable to torch.nn.Linear in format.
    
    This class implements basic methods and parameters that are shared among all version of Linear layers.
    By inhereting from this class one can easily change voltage update, trace update and forward functionalities. 
    """
    def __init__(self,
                 in_features,
                 out_features,
                 batch_size,
                 dt,
                 delay):
        # Dimensions
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size

        self.synapse_shape = (batch_size, out_features, in_features)
        super(_Linear, self).__init__(self.synapse_shape, dt, delay)

        # Learnable parameters
        self.weight = Parameter(torch.Tensor(out_features, in_features))

    # Support function
    def unfold(self, x):
        r"""Placeholder for possible folding functionality."""
        return x

    def fold(self, x):
        r"""Simply folds incoming trace or activation potentials to output format."""
        return x.view(self.batch_size, -1, self.out_features, self.in_features)  # TODO: Add posibility for a channel dim at dim 2

    def update_trace(self, t_in):
        r"""Propagate traces incoming from pre-synaptic neuron through all its outgoing connections."""
        # TODO: Unsure if this clone is needed or not. Might even have to use repeat()
        self.trace.copy_(t_in.expand(-1, self.out_features, -1).contiguous())



class Linear(_Linear):
    r"""SNN linear (fully connected) layer with interface comparable to torch.nn.Linear."""
    def __init__(self,
                 in_features,
                 out_features,
                 batch_size,
                 dt,
                 delay,
                 tau_t,
                 alpha_t):
        super(Linear, self).__init__(in_features, out_features, batch_size, dt, delay)

        # Fixed parameters
        self.tau_t = Parameter(torch.tensor(tau_t, dtype=torch.float))
        self.alpha_t = Parameter(torch.tensor(alpha_t, dtype=torch.float))

        # Initialize connection
        self.init_connection()

    # Standard functions
    # def update_trace(self, x):
    #     r"""Update trace according to exponential decay function and incoming spikes."""
    #     self.trace = sF._linear_trace_update(self.trace, x, self.alpha_t, self.tau_t)

    def activation_potential(self, x):
        r"""Determine activation potentials from each synapse for current time step."""
        out = x * self.weight
        return self.fold(out)

    def forward(self, x, trace_in):
        x = self.convert_spikes(x)
        self.update_trace(trace_in)
        x = self.propagate_spike(x)
        return self.activation_potential(x), self.fold(self.trace)


class LinearExponential(_Linear):
    r"""SNN linear (fully connected) layer with interface comparable to torch.nn.Linear."""
    def __init__(self,
                 in_features,
                 out_features,
                 batch_size,
                 dt,
                 delay,
                 tau_t,
                 alpha_t):
        super(LinearExponential, self).__init__(in_features, out_features, batch_size, dt, delay)

        # Fixed parameters
        self.tau_t = Parameter(torch.tensor(tau_t, dtype=torch.float))
        self.alpha_t = Parameter(torch.tensor(alpha_t, dtype=torch.float))

        # Initialize connection
        self.init_connection()

    # Standard functions
    # def update_trace(self, x):
    #     r"""Update trace according to exponential decay function and incoming spikes."""
    #     self.trace = sF._exponential_trace_update(self.trace, x, self.alpha_t, self.tau_t,
    #                                  self.dt)

    def activation_potential(self, x):
        r"""Determine activation potentials from each synapse for current time step."""
        out = x * self.weight
        # out = x @ self.weight
        # out = x @ self.weight, for automatically summing trace dimension
        return self.fold(out)

    def forward(self, x, trace_in):
        x = self.convert_spikes(x)
        self.update_trace(trace_in)
        x = self.propagate_spike(x)
        return self.activation_potential(x), self.fold(self.trace)


#########################################################
# Convolutional Layers
#########################################################
# Base class
class _ConvNd(Connection):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 im_dims,
                 batch_size,
                 dt,
                 delay,
                 stride=1,
                 padding=0,
                 dilation=1):
        # Assertions
        assert isinstance(im_dims, (tuple, list)), "Parameter im_dims should be a tuple or list of ints"
        for i in im_dims:
            assert isinstance(i, int), "Variables in im_dims should be int"

        # Convolution parameters
        self.batch_size = batch_size  # Cannot infer, needed to reserve memory for storing trace and delay timing
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        # Synapse connections shape
        empty_input = torch.zeros(batch_size, in_channels, *im_dims)
        synapse_shape = list(self.unfold(empty_input).shape)
        synapse_shape[1] = out_channels
        self.synapse_shape = synapse_shape

        # Super init
        super(_ConvNd, self).__init__(synapse_shape, dt, delay)

        # Output image shape
        if len(im_dims) == 1:
            self.image_out_shape = (0)
        elif len(im_dims) == 2:
            self.image_out_shape = conv2d_output_shape(*im_dims, self.kernel_size, stride=self.stride, 
                                                       padding=self.padding, dilation=self.dilation)
        elif len(im_dims) == 3:
            self.image_out_shape = (0, 0, 0)
        else:
            raise ValueError("Input contains too many dimensions")

        # Weight parameter
        self.weight = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.register_parameter("bias", None)

    # Support functions
    def unfold(self, x):
        r"""Simply unfolds incoming image according to layer parameters.
        
        Currently torch.nn.functional.unfold only support 4D tenors (BxCxHxW)!
        """
        # TODO: Possibly implement own folding function that supports 5D if needed
        return F.unfold(x, self.kernel_size, self.dilation, self.padding, self.stride).unsqueeze(1)

    def fold(self, x):
        r"""Simply folds incoming trace or activation potentials according to layer parameters."""
        return x.view(self.batch_size, self.out_channels, *self.image_out_shape, -1)

    def update_trace(self, trace_in):
        self.trace.copy_(trace_in.expand(-1, self.out_channels, -1, -1).contiguous())


class Conv2dExponential(_ConvNd):
    r"""Convolutional SNN layer interface comparable to torch.nn.Conv2d."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 im_dims,
                 batch_size,
                 dt,
                 delay,
                 tau_t,
                 alpha_t,
                 stride=1,
                 padding=0,
                 dilation=1):
        super(Conv2dExponential, self).__init__(in_channels, out_channels, kernel_size, im_dims, batch_size,
                                                dt, delay, stride, padding, dilation)

        # Fixed parameters
        self.tau_t = Parameter(torch.tensor(tau_t, dtype=torch.float))
        self.alpha_t = Parameter(torch.tensor(alpha_t, dtype=torch.float))

        # Intialize layer
        self.init_connection()

    def activation_potential(self, x):
        r"""Determine activation potentials from each synapse for current time step."""
        x = x * self.weight.view(self.weight.shape[0], -1).unsqueeze(2)
        return self.fold(x)

    # Standard functions
    # def update_trace(self, x):
    #     r"""Update trace according to exponential decay function and incoming spikes."""
    #     self.trace = sF._exponential_trace_update(self.trace, x, self.alpha_t, self.tau_t,
    #         self.dt)

    def forward(self, x):
        x = self.convert_spikes(x)
        x = self.unfold(x)  # Till here it is a rather easy set of steps
        self.update_trace(x)
        x = self.propagate_spike(x)  # Output spikes
        return self.activation_potential(x), self.fold(self.trace)


#########################################################
# Pooling Layers
#########################################################
class MaxPool2d(_MaxPoolNd):
    r"""Simple port of PyTorch MaxPool2d with small adjustment for spiking operations.
    
    Currently pooling only supports operations on floating point numbers, thus it casts the uint8 spikes to floats back and forth.
    """
    def reset_state(self):
        pass

    def forward(self, x):
        x = x.to(torch.float32, non_blocking=True)
        x = F.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices)
        return x > 0


class AdaptiveMaxPool2d(_AdaptiveMaxPoolNd):
    r"""Simple port of PyTorch AdaptiveMaxPool2d with small adjustment for spiking operations.
    
    Currently pooling only supports operations on floating point numbers, thus it casts the uint8 spikes to floats back and forth.
    """
    def reset_state(self):
        pass

    def forward(self, x):
        x = x.to(torch.float32, non_blocking=True)
        x = F.adaptive_max_pool2d(x, self.output_size, self.return_indices)
        return x > 0
