"""
Approach correlation based learning, gradient-based learning uses PyTorch's learning rules.
    1. Base LR loops over all layers and provides access to presyn neuron, postsyn neuron, and connection objects.
    2. Applies consistent update function to each layer. This method is variable and can use any of the previously metnioned objects.
    3. Implement a hook mechanism that can modulate the update, i.e. for RL based learning rules.
"""

from collections import OrderedDict
import numpy as np
import torch
from .connection import BaseConnection, _Linear, _ConvNd


#########################################################
# Learning rule base class
#########################################################
class LearningRule:
    r"""Base class for correlation based learning rules in spiking neural networks.
    
    :param layers: An iterable or :class:`dict` of :class:`dict` 
        the latter is a dict that contains a :class:`pysnn.Connection` state dict, a pre-synaptic :class:`pysnn.Neuron` state dict, 
        and a post-synaptic :class:`pysnn.Neuron` state dict that together form a single layer. These objects their state's will be 
        used for optimizing weights.
        During initialization of a learning rule that inherits from this class it is supposed to select only the parameters it needs
        from these objects.
        The higher lever iterable or :class:`dict` contain groups that use the same parameter during training. This is analogous to
        PyTorch optimizers parameter groups.
    :param defaults: A dict containing default hyper parameters. This is a placeholder for possible changes later on, these groups would work
        exactly the same as those for PyTorch optimizers.
    """

    def __init__(self, layers, defaults):
        self.layers = layers
        self.defaults = defaults

    def update_state(self):
        r"""Update state parameters of LearningRule based on latest network forward pass."""
        pass

    def reset_state(self):
        r"""Reset state parameters of LearningRule."""
        pass

    def weight_update(self, layer, params, *args, **kwargs):
        raise NotImplementedError("Each learning rule needs an update function")

    def step(self, *args, **kwargs):
        r"""Performs single learning step for each layer."""
        for l in self.layers.values():
            self.weight_update(l, self.defaults, args, kwargs)

    def pre_mult_post(self, pre, post, conn):
        r"""Multiply a presynaptic term with a postsynaptic term, in the following order: pre x post.

        The outcome of this operation preserves batch size, but furthermore is directly broadcastable 
        with the weight of the connection.

        This operation differs for Linear or Convolutional connections. 

        :param pre: Presynaptic term
        :param post: Postsynaptic term
        :param conn: Connection, support Linear and Conv2d

        :return: Tensor broadcastable with the weight of the connection
        """
        # Select target datatype
        if pre.dtype == torch.bool:
            pre = pre.to(post.dtype)
        elif post.dtype == torch.bool:
            post = post.to(pre.dtype)
        elif pre.dtype != post.dtype:
            raise TypeError(
                "The pre and post synaptic terms should either be of the same datatype, or one of them has to be a Boolean."
            )

        # Perform actual multiplication
        if isinstance(conn, _Linear):
            pre = pre.transpose(2, 1)
        elif isinstance(conn, _ConvNd):
            pre = pre.transpose(2, 1)
            post = post.view(post.shape[0], 1, post.shape[1], -1)
        else:
            if isinstance(conn, BaseConnection):
                raise TypeError(f"Connection type {conn} is not supported.")
            else:
                raise TypeError("Provide an instance of BaseConnection.")

        output = pre * post
        return output.transpose(2, 1)

    def reduce_connections(self, tensor, conn, red_method=torch.mean):
        r"""Reduces the tensor along the dimensions that represent seperate connections to an element of the weight Tensor.

        The function used for reducing has to be a callable that can be applied to single axes of a tensor.
        
        This operation differs or Linear or Convolutional connections.
        For Linear, only the batch dimension (dim 0) is reduced.
        For Conv2d, the batch (dim 0) and the number of kernel multiplications dimension (dim 3) are reduced.

        :param tensor: Tensor that will be reduced
        :param conn: Connection, support Linear and Conv2d
        :param red_method: Method used to reduce each dimension

        :return: Reduced Tensor
        """
        if isinstance(conn, _Linear):
            output = red_method(tensor, dim=0)
        elif isinstance(conn, _ConvNd):
            output = red_method(tensor, dim=(0, 3))
        else:
            if isinstance(conn, BaseConnection):
                raise TypeError(f"Connection type {conn} is not supported.")
            else:
                raise TypeError("Provide an instance of BaseConnection.")

        return output


#########################################################
# STDP
#########################################################
class OnlineSTDP(LearningRule):
    r"""Basic online STDP implementation from http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity

    :param layers: OrderedDict containing state dicts for each layer.
    :param lr: Learning rate.
    """

    def __init__(
        self, layers, lr=0.001, a_plus=1.0, a_min=1.0,
    ):
        params = dict(lr=lr, a_plus=a_plus, a_min=a_min)
        super(OnlineSTDP, self).__init__(layers, params)

    def weight_update(self, layer, params, *args, **kwargs):
        pre_trace, post_trace = layer.presynaptic.trace, layer.postsynaptic.trace
        pre_spike, post_spike = layer.presynaptic.spikes, layer.postsynaptic.spikes
        dw = params["a_plus"] * self.pre_mult_post(
            pre_trace, post_spike, layer.connection
        )
        dw = params["a_plus"] * self.pre_mult_post(
            pre_spike, post_trace, layer.connection
        )
        layer.connection.weight += params["lr"] * self.reduce_connections(
            dw, layer.connection
        )


#########################################################
# MSTDPET
#########################################################
class MSTDPET(LearningRule):
    r"""Apply MSTDPET from (Florian 2007) to the provided connections.
    
    Uses just a single, scalar reward value.
    Update rule can be applied at any desired time step.

    :param layers: OrderedDict containing state dicts for each layer.
    :param a_pre: Scaling factor for presynaptic spikes influence on the eligibilty trace.
    :param a_post: Scaling factor for postsynaptic spikes influence on the eligibilty trace.
    :param lr: Learning rate.
    :param e_trace_decay: Decay factor for the eligibility trace.
    """

    def __init__(self, layers, a_pre, a_post, lr, e_trace_decay):
        self.check_layers(layers)

        # Collect desired tensors from state dict in a layer object
        for key, layer in layers.items():
            new_layer = {}
            new_layer["pre_spikes"] = layer["connection"]["spikes"]
            new_layer["pre_trace"] = layer["connection"]["trace"]
            new_layer["post_spikes"] = layer["neuron"]["spikes"]
            new_layer["post_trace"] = layer["neuron"]["trace"]
            new_layer["weight"] = layer["connection"]["weight"]
            new_layer["e_trace"] = torch.zeros_like(layer["connection"]["trace"])
            new_layer["type"] = layer["type"]
            layers[key] = new_layer

        self.a_pre = a_pre
        self.a_post = a_post
        self.lr = lr
        self.e_trace_decay = e_trace_decay

        # To possibly later support groups, without changing interface
        defaults = {
            "a_pre": a_pre,
            "a_post": a_post,
            "lr": lr,
            "e_trace_decay": e_trace_decay,
        }

        super(MSTDPET, self).__init__(layers, defaults)

    def update_state(self):
        r"""Update eligibility trace based on pre and postsynaptic spiking activity.
        
        This function has to be called manually at desired times, often after each timestep.
        """

        for layer in self.layers.values():
            # Update eligibility trace
            layer["e_trace"] *= self.e_trace_decay
            layer["e_trace"] += self.a_pre * self.pre_mult_post(
                layer["pre_trace"], layer["post_spikes"], layer["type"]
            )
            layer["e_trace"] -= self.a_post * self.pre_mult_post(
                layer["pre_spikes"], layer["post_trace"], layer["type"]
            )

    def reset_state(self):
        for layer in self.layers.values():
            layer["e_trace"].fill_(0)

    def step(self, reward):
        r"""Performs single learning step.
        
        :param reward: Scalar reward value.
        """

        # TODO: add weight clamping?
        for layer in self.layers.values():
            dw = self.reduce_connections(layer["e_trace"], layer["type"])
            layer["weight"] += self.lr * reward * dw.view(*layer["weight"].shape)


#########################################################
# Fede STDP
#########################################################
class FedeSTDP(LearningRule):
    r"""STDP version for Paredes Valles, performs mean operation over the batch dimension before weight update.

    Defined in "Unsupervised Learning of a Hierarchical Spiking Neural Network for Optical Flow Estimation: From Events to Global Motion Perception - F.P. Valles, et al."

    :param layers: OrderedDict containing state dicts for each layer.
    :param lr: Learning rate.
    :param w_init: Initialization/reference value for all weights.
    :param a: Stability parameter, a < 1.
    """

    def __init__(self, layers, lr, w_init, a):
        assert lr > 0, "Learning rate should be positive."
        assert (a <= 1) and (a >= 0), "For FedeSTDP 'a' should fall between 0 and 1."

        # Check layer formats
        self.check_layers(layers)

        # Set default hyper parameters
        self.lr = lr
        self.w_init = w_init
        self.a = a

        # To possibly later support groups, without changing interface
        defaults = {"lr": lr, "w_init": w_init, "a": a}

        # Select only necessary parameters
        for key, layer in layers.items():
            new_layer = {}
            new_layer["trace"] = layer["connection"]["trace"]
            new_layer["weight"] = layer["connection"]["weight"]
            layers[key] = new_layer

        super(FedeSTDP, self).__init__(layers, defaults)

    def step(self):
        r"""Performs single learning step."""
        for layer in self.layers.values():
            w = layer["weight"]

            # Normalize trace
            trace = layer["trace"].view(-1, *w.shape)
            norm_trace = trace / trace.max()

            # LTP and LTD
            dw = w - self.w_init

            # LTP computation
            ltp_w = torch.exp(-dw)
            ltp_t = torch.exp(norm_trace) - self.a
            ltp = ltp_w * ltp_t

            # LTD computation
            ltd_w = -(torch.exp(dw))
            ltd_t = torch.exp(1 - norm_trace) - self.a
            ltd = ltd_w * ltd_t

            # Perform weight update
            layer["weight"] += self.lr * (ltp + ltd).mean(0)
