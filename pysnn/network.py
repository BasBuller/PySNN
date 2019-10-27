from collections import OrderedDict
import torch
import torch.nn as nn

from pysnn.connection import Connection
from pysnn.neuron import BaseNeuron


#########################################################
# SNN Network
#########################################################
class SNNNetwork(nn.Module):
    r"""Simple base clase for defining SNN network, contains some convenience operators
    for e.g. clearing network state after simulating a sample.
    """

    def __init__(self):
        super(SNNNetwork, self).__init__()
        self._layers = OrderedDict()

    def reset_state(self):
        for module in self.modules():
            if not isinstance(module, SNNNetwork):
                module.reset_state()

    def add_layer(self, name, connection, neuron):
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

        # Add layer
        self._layers[name] = {"connection": connection, "neuron": neuron}

    def layer_state_dict(self):
        r"""Return state dicts grouped per layer, so a single Connection and a single Neuron state dict per layer."""
        state_dicts = OrderedDict()
        for name, layer in self._layers.items():
            states = {k: v.state_dict() for k, v in layer.items()}
            state_dicts[name] = states
        return state_dicts
