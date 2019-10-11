from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from pysnn.connection import Connection
from pysnn.utils import _set_no_grad, tensor_clamp


#########################################################
# Learning rule base class
#########################################################
class LearningRule:
    r"""Base class for correlation based learning rules in spiking neural networks.
    
    Arguments:
        layers (iterable): an iterable or :class:`dict` of :class:`dict`s. 
            The latter is a dict that contains a :class:`pysnn.Connection`s state dict, a pre-synaptic :class:`pysnn.Neuron`s state dict, 
            and a post-synaptic :class:`pysnn.Neuron`s state dict that together form a single layer. These objects their state's will be 
            used for optimizing weights.
            During initialization of a learning rule that inherits from this class it is supposed to select only the paramters it needs
            from these objects.
            The higher lever iterable or :class:`dict` contain groups that use the same parameter during training. This is analogous to
            PyTorch optimizers' parameter groups.
        defaults: A dict containing default hyper paramters. This is a placeholder for possible changes later on, these groups would work
            exactly the same as those for PyTorch optimizers.
    """

    def __init__(self, layers, defaults):
        self.defaults = defaults
        self.layers = layers

    def update_state(self):
        r"""Update state parameters of LearningRule based on latest network forward pass."""
        raise NotImplementedError

    def step(self):
        r"""Performs single learning step."""
        raise NotImplementedError

    def reset_state(self):
        raise NotImplementedError

    def add_layer_group(self, layer):
        pass

    def check_layers(self, layers):
        r"""Check if layers provided to constructor are of the right format."""

        # Check if layers is iterator
        if not isinstance(layers, (list, tuple)):
            raise TypeError(
                "Layers should be an iterator with deterministic ordering, a list or a tuple. Current type is "
                + type(layers)
            )

        # Check for empty iterator
        if len(layers) == 0:
            raise ValueError("Got an empty layers iterator.")

        # Check for type of layers
        if not isinstance(layers[0], dict):
            raise TypeError(
                "A layer object should be a dict. Currently got a " + type(layers[0])
            )


#########################################################
# MSTDPET
#########################################################
class MSTDPET(LearningRule):
    r"""Apply MSTDPET from (Florian 2007) to the provided connections.
    
    Uses just a single, scalar reward value.
    Update rule can be applied at any desired time step.
    """

    def __init__(
        self, layers, a_pre=1, a_post=1, lr=0.0001, e_trace_decay=float(np.exp(-1 / 20))
    ):
        self.check_layers(layers)

        # Collect desired tensors from state dict in a layer object
        for idx, layer in enumerate(layers):
            new_layer = {}
            new_layer["pre_spikes"] = layer["pre_neuron"]["spikes"]
            new_layer["pre_trace"] = layer["pre_neuron"]["trace"]
            new_layer["post_spikes"] = layer["post_neuron"]["spikes"]
            new_layer["post_trace"] = layer["post_neuron"]["trace"]
            new_layer["weight"] = layer["connection"]["weight"]
            new_layer["e_trace"] = torch.zeros_like(layer["connection"]["weight"])
            layers[idx] = new_layer

        self.a_pre = a_pre
        self.a_post = a_post
        self.lr = lr
        self.e_trace_decay = e_trace_decay

        # To possibly later support groups, without chaning interface
        defaults = {
            "a_pre": a_pre,
            "a_post": a_post,
            "lr": lr,
            "e_trace_decay": e_trace_decay,
        }

        super(MSTDPET, self).__init__(layers, defaults)

    def update_state(self):
        r"""Update eligibility trace based on pre and postsynaptic spiking activity.
        
        This function has to be called manually after each timestep. Should not be called from within forward, 
        as this does is likely not called every timestep.
        """

        for layer in self.layers:
            # Update eligibility trace
            layer["e_trace"] *= self.e_trace_decay
            layer["e_trace"] += self.a_pre * torch.ger(
                layer["post_spikes"].view(-1).float(), layer["pre_trace"].view(-1)
            ) - self.a_post * torch.ger(
                layer["post_trace"].view(-1), layer["pre_spikes"].view(-1).float()
            )

    def reset_state(self):
        for layer in self.layers:
            layer["e_trace"].fill_(0)

    def step(self, reward):
        # TODO: add weight clamping?
        for layer in self.layers:
            layer["weight"] += self.defaults["lr"] * reward * layer["e_trace"]


#########################################################
# Fede STDP
#########################################################
class FedeSTDP(LearningRule):
    r"""STDP version for Paredes Valles, performs mean operation over the batch 
    dimension before weight update."""

    def __init__(self, layers, lr, w_init, a):
        # Check layer formats
        self.check_layers(layers)

        # Set default hyper parameters
        self.lr = lr
        self.w_init = w_init
        self.a = a

        # To possibly later support groups, without chaning interface
        defaults = {"lr": lr, "w_init": w_init, "a": a}

        # Select only necessary paramters
        for idx, layer in enumerate(layers):
            new_layer = {}
            new_layer["trace"] = layer["connection"]["trace"]
            new_layer["weight"] = layer["connection"]["weight"]
            layers[idx] = new_layer

        super(FedeSTDP, self).__init__(layers, defaults)

    def step(self):
        for layer in self.layers:
            w = layer["weight"]
            trace = layer["trace"].view(-1, *w.shape)

            # LTP and LTD
            dw = w - self.w_init

            # LTP computation
            ltp_w = torch.exp(-dw)
            ltp_t = torch.exp(trace) - self.a
            ltp = ltp_w * ltp_t

            # LTD computation
            ltd_w = -(torch.exp(dw))
            ltd_t = torch.exp(1 - trace) - self.a
            ltd = ltd_w * ltd_t

            # Perform weight update
            layer["weight"] += self.lr * (ltp + ltd).mean(0)
