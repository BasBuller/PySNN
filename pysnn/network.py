from collections import OrderedDict
import torch
import torch.nn as nn


#########################################################
# Utilities
#########################################################
def _tag_tensor(item, tag):
    if isinstance(item, torch.Tensor):  # TODO: unsure if this is sufficient
        item._parent = tag
    

def _graph_tracing_pre_hook(self, input):
    r"""Trace computational graph of a SNN using _forward_pre_hooks.
    
    Stores parent node in self._prev. The previous object location is stored in incoming Tensor.
    """
    prevs = [i._parent for i in input if hasattr(i, "_parent") and i._parent]
    self._prev = set(prevs)


def _graph_tracing_post_hook(self, _, output):
    r"""Tag module outputs with the object that produced them."""
    for i in output:
        _tag_tensor(i, self)


#########################################################
# SNN Module
#########################################################
class SpikingModule(nn.Module):
    def __init__(self):
        super(SpikingModule, self).__init__()

        # Keep track of spiking neuron specific elements
        self._layers = OrderedDict()
        self._neurons = OrderedDict()
        self._connections = OrderedDict()

        # For easy tree traversal
        self.name = None
        self.prev_module = []
        self.next_module = []
        self._prev = set()


    def __setattr__(self, name, value):
        r"""Performs tracking of neurons and connections, in addition to nn.Module tracking of PyTorch."""

        super(SpikingModule, self).__setattr__(name, value)

        if isinstance(value, SpikingModule):
            prefix = self.name + "." if self.name else ""
            value.name = prefix + name
            for con in value._connections.values():
                con.name = value.name + "." + con.name
            for neur in value._neurons.values():
                neur.name = value.name + "." + neur.name
            if value._layers:
                self._layers[name] = value


    # ######################################################
    # Spiking versions of regular utility functions
    # ######################################################
    def named_spiking_children(self):
        r"""Regular named_children function only for SpikingModules."""
        memo = set()
        for name, module in self._modules.items():
            if module is not None and isinstance(module, SpikingModule) and module not in memo:
                memo.add(module)
                yield name, module

    
    def spiking_children(self):
        r"""Regular children function only for SpikingModules."""
        for _, module in self.named_spiking_children():
            yield module

    
    def named_spiking_modules(self, memo=None, prefix=""):
        r"""Regular named modules function only for SpikingModules."""
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self

            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                for m in module.named_spiking_modules(memo, submodule_prefix):
                    yield m


    def spiking_modules(self):
        r"""Regular modules function only for SpikingModules."""
        for _, module in self.named_spiking_modules():
            yield module


    def spiking_apply(self, fn):
        r"""Regular apply function only for SpikingModules."""
        for module in self.spiking_children():
            module.spiking_apply(fn)
        fn(self)
        return self


    # ######################################################
    # Graph tracing
    # ######################################################
    def trace_graph(self, *input, **kwargs):
        r"""Trace complete graph by tracking the flow of data through the network."""
        pre_hooks, hooks = {}, {}
        for name, module in self.named_spiking_modules():
            pre_hooks[name] = module.register_forward_pre_hook(_graph_tracing_pre_hook)
            hooks[name] = module.register_forward_hook(_graph_tracing_post_hook)

        for i in input:
            _tag_tensor(i, ())

        # Forward tracing of input
        out = self.forward(*input, **kwargs)
        out = out[0] if isinstance(out, (tuple, list)) else out  # select just single output tensor

        # Construct graph from output to input
        nodes, edges, topo = set(), set(), []
        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)
                topo.append(v)
        build(out._parent)

        # Clean hooks, TODO: Make select for just the added hooks
        for mod in self.spiking_modules():
            mod._forward_pre_hooks.clear()
            mod._forward_hooks.clear()
        self.reset_state()

        return nodes, edges, topo


    # ######################################################
    # Spiking functions
    # ######################################################
    def reset_state(self):
        r"""Resets state parameters of all submodules, requires each submodule to have a reset_state function.
        
        When defining reset_state at network level, make sure to call super().reset_state() at the end of it.
        This makes sure that all sub SpikingModules.reset_state() is called.
        """
        for module in self.spiking_modules():
            if module is not self:
                module.reset_state()


    def _save_to_layer_state_dict(self, layer, modules, keep_vars):
        dict_names = ["connection", "neuron", "presyn_neuron"]
        states = {}
        for name, value in layer.items():
            if name != "type":
                state = value.state_dict(keep_vars=keep_vars)
                state["name"] = value.name
                states[name] = state
            elif name == "type":
                states[name] = value
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
