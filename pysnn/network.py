from collections import OrderedDict
import torch
import torch.nn as nn

from pysnn.connection import Connection, Linear, Conv2d
from pysnn.neuron import BaseNeuron, BaseInput


#########################################################
# SNN Network
#########################################################
class SNNNetwork(nn.Module):
    r"""Base clase for defining SNN network, contains several convenience operators for network simulations.
    """

    def __init__(self):
        super(SNNNetwork, self).__init__()
        self._layers = OrderedDict()

    def reset_state(self):
        r"""Resets state parameters of all submodules, requires each submodule to have a reset_state function."""
        for module in self.modules():
            if not isinstance(module, SNNNetwork):
                module.reset_state()

    def add_layer(self, name, connection, neuron):
        r"""Adds which :class:`Neuron` and :class:`Connection` objects together form a layer, as well as the layer type.
        
        :param name: Name of the layer
        :param connection: :class:`Connection` object
        :param neuron: :class:`Neuron` object
        """
        # Check connection object
        if not isinstance(connection, Connection):
            raise TypeError("Connection input needs to be a Connection object.")

        # Check neuron object
        if not isinstance(neuron, BaseNeuron):
            raise TypeError("Neuron input needs to be a BaseNeuron object.")

        # Check name
        if name in self._layers:
            raise KeyError("Layer name already exists, please use a different one.")
        elif "." in name:
            raise KeyError("Name cannot contain  a '.'.")
        elif name == "":
            raise KeyError("Name cannot be an empty string.")

        # Check specific connection type
        if isinstance(connection, Linear):
            ctype = "linear"
        elif isinstance(connection, Conv2d):
            ctype = "conv2d"
        else:
            raise TypeError("Connection is of an unkown type.")

        # Add layer
        self._layers[name] = {"connection": connection, "neuron": neuron, "type": ctype}

    def layer_state_dict(self):
        r"""Return state dicts grouped per layer, so a single Connection and a single Neuron state dict per layer.
        
        :return: State Dicts for the :class:`Connection` and :class:`Neuron` of the layer, as well as the layer type.
        """
        dict_names = ["connection", "neuron"]
        state_dicts = OrderedDict()
        for name, layer in self._layers.items():
            states = {}
            for k, v in layer.items():
                if k in dict_names:
                    states[k] = v.state_dict()
                elif k == "type":
                    states[k] = v
            state_dicts[name] = states
        return state_dicts

    def change_batch_size(self, batch_size):
        r"""Changes the batch dimension of all state tensors. Be careful, only call this method after resetting state, otherwise part of your data will be lost."""

        for module in self.modules():
            if isinstance(module, (BaseNeuron, BaseInput, Connection)):
                module.change_batch_size(batch_size)
