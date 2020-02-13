from collections import OrderedDict
import torch
import torch.nn as nn

from pysnn.connection import Connection, _Linear, _ConvNd, _Recurrent
from pysnn.neuron import BaseNeuron, BaseInput


#########################################################
# SNN Module
#########################################################
class SpikingModule(nn.Module):
    def __init__(self):
        super(SpikingModule, self).__init__()
        self._layers = OrderedDict()

    def reset_state_recursive(self):
        r"""Reset state of all child nodes of this module."""
        for module in self._modules.values():
            module.reset_state()

    def reset_state(self):
        r"""Resets state parameters of all submodules, requires each submodule to have a reset_state function."""
        # raise NotImplementedError("Every SNN Module needs to implement a reset state function.")
        self.reset_state_recursive()

    def add_layer(self, name, connection, neuron, presyn_neuron=None):
        r"""Adds which :class:`Neuron` and :class:`Connection` objects together form a layer, as well as the layer type.
        
        :param name: Name of the layer
        :param connection: :class:`Connection` object
        :param neuron: postsynaptic :class:`Neuron` object, required
        :param presyn_neuron: Optional, presynaptic :class:`Neuron` object
        """
        # Check connection object
        if not isinstance(connection, (Connection, str)):
            raise TypeError("Connection input needs to be a Connection object.")

        # Check neuron object
        if not isinstance(neuron, (BaseNeuron, str)):
            raise TypeError("Neuron input needs to be a BaseNeuron object.")

        # Check presynaptic neuron, if supplied:
        if presyn_neuron is not None:
            if not isinstance(neuron, (BaseNeuron, BaseInput, str)):
                raise TypeError(
                    "Presynaptic neuron needs to be a BaseNeuron or BaseInput object."
                )

        # Check name
        if name in self._layers:
            raise KeyError("Layer name already exists, please use a different one.")
        elif "." in name:
            raise KeyError("Name cannot contain  a '.'")
        elif name == "":
            raise KeyError("Name cannot be an empty string.")

        # Check specific connection type
        if isinstance(
            self._modules[connection] if isinstance(connection, str) else connection,
            _Linear,
        ):
            ctype = "linear"
        elif isinstance(
            self._modules[connection] if isinstance(connection, str) else connection,
            _ConvNd,
        ):
            ctype = "conv2d"
        elif isinstance(
            self._modules[connection] if isinstance(connection, str) else connection,
            _Recurrent,
        ):
            ctype = "recurrent"
        else:
            raise TypeError("Connection is of an unkown type.")

        # Add layer
        self._layers[name] = {"connection": connection, "neuron": neuron, "type": ctype}
        if presyn_neuron:
            self._layers[name]["presyn_neuron"] = presyn_neuron

    def add_module_layer(self, module_name):
        assert (
            module_name in self._modules
        ), "Name of snn modules has to be present in modules list."
        self._layers[module_name] = self._modules[module_name]

    def _save_to_layer_state_dict(self, layer, modules, keep_vars):
        dict_names = ["connection", "neuron", "presyn_neuron"]
        states = {}
        for k, v in layer.items():
            if k in dict_names:
                obj_name = None
                if isinstance(v, str):
                    obj_name = v
                    v = modules[
                        v
                    ]  # TODO: Change this layer object to the actual modules it is referring to!
                states[k] = v.state_dict(keep_vars=keep_vars)
                if obj_name:
                    states[k]["name"] = obj_name
            elif k == "type":
                states[k] = v

        return states

    def layer_state_dict(
        self, destination=None, prefix="", modules=None, keep_vars=False
    ):
        # Ordered dict containing all subsequent state dicts
        if destination is None:
            destination = OrderedDict()

        if modules is None:
            modules = self._modules

        # Loop over layers and add possible sublayers
        for layer_name, layer in self._layers.items():

            # Recursively iterator further if spiking module
            if isinstance(layer, SpikingModule) and layer._layers:
                sub_modules = modules[layer_name]._modules
                layer.layer_state_dict(
                    destination=destination,
                    prefix=prefix + layer_name + ".",
                    modules=sub_modules,
                    keep_vars=keep_vars,
                )

            # If layer, just add to output dict
            elif isinstance(layer, dict):
                destination[prefix + layer_name] = self._save_to_layer_state_dict(
                    layer, modules, keep_vars
                )

        return destination

    def change_batch_size(self, batch_size):
        r"""Changes the batch dimension of all state tensors. Be careful, only call this method after resetting state, otherwise part of your data will be lost.
        
        returns batch size from before adjusting it.
        """

        cur_bsize = None
        for module in self.modules():
            if isinstance(module, BaseInput) and not cur_bsize:
                cur_bsize = module.return_batch_size()
            if isinstance(module, (BaseNeuron, BaseInput, Connection)):
                module.change_batch_size(batch_size)
        return cur_bsize


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
        for module in self._modules.values():
            if not isinstance(module, SNNNetwork):
                module.reset_state()

    def add_layer(self, name, connection, neuron, presyn_neuron=None):
        r"""Adds which :class:`Neuron` and :class:`Connection` objects together form a layer, as well as the layer type.
        
        :param name: Name of the layer
        :param connection: :class:`Connection` object
        :param neuron: postsynaptic :class:`Neuron` object, required
        :param presyn_neuron: Optional, presynaptic :class:`Neuron` object
        """
        # Check connection object
        if not isinstance(connection, (Connection, str)):
            raise TypeError("Connection input needs to be a Connection object.")

        # Check neuron object
        if not isinstance(neuron, (BaseNeuron, str)):
            raise TypeError("Neuron input needs to be a BaseNeuron object.")

        # Check presynaptic neuron, if supplied:
        if presyn_neuron is not None:
            if not isinstance(neuron, (BaseNeuron, BaseInput, str)):
                raise TypeError(
                    "Presynaptic neuron needs to be a BaseNeuron or BaseInput object."
                )

        # Check name
        if name in self._layers:
            raise KeyError("Layer name already exists, please use a different one.")
        elif "." in name:
            raise KeyError("Name cannot contain  a '.'")
        elif name == "":
            raise KeyError("Name cannot be an empty string.")

        # Check specific connection type
        if isinstance(
            self._modules[connection] if isinstance(connection, str) else connection,
            _Linear,
        ):
            ctype = "linear"
        elif isinstance(
            self._modules[connection] if isinstance(connection, str) else connection,
            _ConvNd,
        ):
            ctype = "conv2d"
        elif isinstance(
            self._modules[connection] if isinstance(connection, str) else connection,
            _Recurrent,
        ):
            ctype = "recurrent"
        else:
            raise TypeError("Connection is of an unkown type.")

        # Add layer
        self._layers[name] = {"connection": connection, "neuron": neuron, "type": ctype}
        if presyn_neuron:
            self._layers[name]["presyn_neuron"] = presyn_neuron

    def layer_state_dict(self, keep_vars=False):
        r"""Return state dicts grouped per layer, so a single Connection and a single Neuron state dict per layer.
        
        :return: State Dicts for the :class:`Connection` and :class:`Neuron` of the layer, as well as the layer type.
        """
        dict_names = ["connection", "neuron", "presyn_neuron"]
        state_dicts = OrderedDict()
        for layer_name, layer in self._layers.items():
            states = {}
            for k, v in layer.items():
                if k in dict_names:
                    obj_name = None
                    if isinstance(v, str):
                        obj_name = v
                        v = self._modules[v]
                    states[k] = v.state_dict(keep_vars=keep_vars)
                    if obj_name:
                        states[k]["name"] = obj_name
                elif k == "type":
                    states[k] = v
            state_dicts[layer_name] = states
        return state_dicts

    def change_batch_size(self, batch_size):
        r"""Changes the batch dimension of all state tensors. Be careful, only call this method after resetting state, otherwise part of your data will be lost.
        
        returns batch size from before adjusting it.
        """

        cur_bsize = None
        for module in self.modules():
            if isinstance(module, BaseInput) and not cur_bsize:
                cur_bsize = module.return_batch_size()
            if isinstance(module, (BaseNeuron, BaseInput, Connection)):
                module.change_batch_size(batch_size)
        return cur_bsize
