import numpy as np
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

    def __init__(self, shape, dt, delay):
        super(Connection, self).__init__()
        self.synapse_shape = torch.tensor(shape)

        # TODO: Should there be a check for dt fitting integer number of times in delay duration?

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
            raise TypeError(
                "Incorrect data type provided for delay_init, please provide an int or FloatTensor"
            )
        self.register_buffer("delay_init", delay_init)

        # Fixed parameters
        self.register_buffer("dt", torch.tensor(dt, dtype=torch.float))

        # State parameters
        self.register_buffer("trace", torch.Tensor(*shape))
        self.register_buffer("delay", torch.Tensor(*shape))
        self.register_buffer("spikes", torch.Tensor(*shape).bool())

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
        self.spikes.fill_(False)

    def reset_weights(self, distribution="uniform", gain=1.0, a=0.0, b=1.0):
        r"""Reinnitialize network weights.

        Note: Not all parameters apply to every distribution.
        
        :param distribution: Distribution used for initialization
        :param gain: Scalar increase of the weight values.
        :param a: Lower bound.
        :param b: Upper bound.
        """
        if distribution == "uniform":
            nn.init.uniform_(self.weight, a=a, b=b)
        elif distribution == "neuron_scaled_uniform":
            scaling = np.sqrt(self.weight.shape[1])
            a = a / scaling
            b = b / scaling
            nn.init.uniform_(self.weight, a=a, b=b)
        elif distribution == "normalized":
            nn.init.uniform_(self.weight, a=a, b=b)
            self.weight /= self.weight.sum()
            self.weight *= gain
        elif distribution == "normal":
            nn.init.normal_(self.weight)
        elif distribution == "xavier_normal":
            nn.init.xavier_normal_(self.weight, gain=gain)
        elif distribution == "xavier_uniform":
            nn.init.xavier_uniform_(self.weight, gain=gain)
        elif distribution == "kaiming_normal":
            nn.init.kaiming_normal_(self.weight)
        elif distribution == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.weight)
        elif distribution == "constant":
            nn.init.constant_(self.weight, gain)

    def init_connection(self):
        r"""Collection of all intialization methods.
        
        Assumes weights are implemented by the class that inherits from this base class.
        """
        assert hasattr(
            self, "weight"
        ), f"Weight attribute is missing for {self.__class__.__name__}."
        assert isinstance(
            self.weight, Parameter
        ), f"Weight attribute is not a PyTorch Parameter for {self.__class__.__name__}."
        self.no_grad()
        self.reset_state()
        self.reset_weights()

    def propagate_spike(self, x):
        r"""Track propagation of spikes through synapses if the connection."""
        if self.delay_init is not None:
            self.delay[self.delay > 0] -= 1
            spike_out = self.delay == 1
            self.delay += self.delay_init * x
        else:
            spike_out = x
        self.spikes.copy_(spike_out)
        return self.convert_spikes(spike_out)

    def change_batch_size(self, batch_size):
        r"""Changes the batch dimension of all state tensors. Be careful, only call this method after resetting state, otherwise part of your data will be lost."""
        # Update to new shape
        self.synapse_shape[0] = batch_size
        n_shape = self.synapse_shape

        self.spikes.resize_(*n_shape)
        self.delay.resize_(*n_shape)
        self.trace.resize_(*n_shape)


#########################################################
# Linear Layers
#########################################################
# Base class
class _Linear(Connection):
    r"""SNN linear base class, comparable to torch.nn.Linear in format.
    
    This class implements basic methods and parameters that are shared among all version of Linear layers.
    By inhereting from this class one can easily change voltage update, trace update and forward functionalities. 
    """

    def __init__(self, in_features, out_features, batch_size, dt, delay):
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
        r"""Fold incoming spike, trace, or activation potentials to output format.
        
        :param x: Tensor containing spikes, traces, or activations.

        :return: Folder input tensor.
        """
        return x.view(
            self.synapse_shape[0], -1, self.out_features, self.in_features
        )  # TODO: Add posibility for a channel dim at dim 2

    def update_trace(self, t_in):
        r"""Propagate traces incoming from pre-synaptic neuron through all its outgoing connections.
        
        :param t_in: Current values of the trace to be stored.
        """
        # TODO: Unsure if this clone is needed or not. Might even have to use repeat()
        self.trace.copy_(t_in.expand(-1, self.out_features, -1).contiguous())


class Linear(_Linear):
    r"""SNN linear (fully connected) layer with interface comparable to torch.nn.Linear.
    
    :param in_features: Size of each input sample.
    :param out_features: Size of each output sample.
    :param batch_size: Number of samples in a batch.
    :param dt: Duration of each timestep.
    :param delay: Time it takes for a spike to propagate through the connection. Should be an integer multiple of dt.
    """

    def __init__(self, in_features, out_features, batch_size, dt, delay):
        super(Linear, self).__init__(in_features, out_features, batch_size, dt, delay)

        # Initialize connection
        self.init_connection()

    def activation_potential(self, x):
        r"""Determine activation potentials from each synapse for current time step.
        
        :param x: Presynaptic spikes.

        :return: Activation potentials.
        """
        out = x * self.weight
        return self.fold(out)

    def forward(self, x, trace_in):
        r"""Calculate postsynaptic activation potentials and trace.
        
        :param x: Presynaptic spikes.
        :param trace_in: Presynaptic trace.

        :return: (Activation potentials, Postsynaptic trace)
        """
        self.update_trace(trace_in)
        x = self.convert_spikes(x)
        x = self.propagate_spike(x)
        return self.activation_potential(x), self.fold(self.trace)


#########################################################
# Convolutional Layers
#########################################################
# Base class
class _ConvNd(Connection):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        im_dims,
        batch_size,
        dt,
        delay,
        stride=1,
        padding=0,
        dilation=1,
    ):
        # Assertions
        assert isinstance(
            im_dims, (tuple, list)
        ), "Parameter im_dims should be a tuple or list of ints."
        for i in im_dims:
            assert isinstance(i, int), "Variables in im_dims should be int."

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
            self.image_out_shape = 0
        elif len(im_dims) == 2:
            self.image_out_shape = conv2d_output_shape(
                *im_dims,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
        elif len(im_dims) == 3:
            self.image_out_shape = (0, 0, 0)
        else:
            raise ValueError("Input contains too many dimensions")

        # Weight parameter
        self.weight = Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size)
        )

    # Support functions
    def unfold(self, x):
        r"""Simply unfolds incoming image according to layer parameters.
        
        Currently torch.nn.functional.unfold only support 4D tenors (BxCxHxW)!
        """
        # TODO: Possibly implement own folding function that supports 5D if needed
        return F.unfold(
            x, self.kernel_size, self.dilation, self.padding, self.stride
        ).unsqueeze(1)

    def fold(self, x):
        r"""Simply folds incoming trace or activation potentials according to layer parameters."""
        return x.view(self.batch_size, self.out_channels, *self.image_out_shape, -1)

    def update_trace(self, trace_in):
        self.trace.copy_(trace_in.expand(-1, self.out_channels, -1, -1).contiguous())


class Conv2d(_ConvNd):
    r"""Convolutional SNN layer interface comparable to torch.nn.Conv2d.
    
    :param: Number of channels in the input image.
    :param out_channels: Number of channels produced by the convolution
    :param kernel_size: Size of the convolving kernel
    :param im_dims: (Height, Width) of the input image.
    :param batch_size: Number of samples in a batch.
    :param dt: Duration of each timestep.
    :param delay: Time it takes for a spike to propagate through the connection. Should be an integer multiple of dt.
    :param stride: Stride of the convolution. Default: 1
    :param padding: Zero-padding added to both sides of the input. Default: 0
    :param dilation: Spacing between kernel elements. Default: 1
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        im_dims,
        batch_size,
        dt,
        delay,
        stride=1,
        padding=0,
        dilation=1,
    ):
        super(Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            im_dims,
            batch_size,
            dt,
            delay,
            stride,
            padding,
            dilation,
        )

        # Intialize layer
        self.init_connection()

    def activation_potential(self, x):
        r"""Determine activation potentials from each synapse for current time step.
        
        :param x: Presynaptic spikes.

        :return: Activation potentials.
        """
        x = x * self.weight.view(self.weight.shape[0], -1).unsqueeze(2)
        return self.fold(x)

    def forward(self, x, trace_in):
        r"""Calculate postsynaptic activation potentials and trace.
        
        :param x: Presynaptic spikes.
        :param trace_in: Presynaptic trace.

        :return: (Activation potentials, Postsynaptic trace)
        """
        trace_in = self.unfold(trace_in)
        self.update_trace(trace_in)
        x = self.convert_spikes(x)
        x = self.unfold(x)  # Till here it is a rather easy set of steps
        x = self.propagate_spike(x)  # Output spikes
        return self.activation_potential(x), self.fold(self.trace)


#########################################################
# Max Pooling
#########################################################
class _SpikeMaxPoolNd(nn.Module):
    def __init__(
        self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False
    ):
        super(_SpikeMaxPoolNd, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = True

    def reset_state(self):
        pass


class MaxPool2d(_SpikeMaxPoolNd):
    r"""Simple port of PyTorch MaxPool2d with small adjustment for spiking operations.
    
    Currently pooling only supports operations on floating point numbers, thus it casts the uint8 spikes to floats back and forth.
    The trace of the 'maximum' spike is also returned. In case of multiple spikes within pooling window, returns first spike of 
    the window (top left corner).
    """

    def forward(self, x, trace):
        x = x.to(torch.float32, non_blocking=True)
        x, idx = F.max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )
        trace = trace.view(-1)[idx.view(-1)]
        trace = trace.view(idx.shape)
        return x > 0, trace


#########################################################
# Adaptive Max Pooling
#########################################################
class _SpikeAdaptiveMaxPoolNd(nn.Module):
    def __init__(self, output_size):
        super(_SpikeAdaptiveMaxPoolNd, self).__init__()
        self.output_size = output_size
        self.return_indices = True

    def reset_state(self):
        pass


class AdaptiveMaxPool2d(_SpikeAdaptiveMaxPoolNd):
    r"""Simple port of PyTorch AdaptiveMaxPool2d with small adjustment for spiking operations.
    
    Currently pooling only supports operations on floating point numbers, thus it casts the uint8 spikes to floats back and forth.
    The trace of the 'maximum' spike is also returned. In case of multiple spikes within pooling window, returns first spike of 
    the window (top left corner).
    """

    def forward(self, x, trace):
        x = x.to(torch.float32, non_blocking=True)
        x, idx = F.adaptive_max_pool2d(x, self.output_size, self.return_indices)
        trace = trace[idx]
        return x > 0, trace
