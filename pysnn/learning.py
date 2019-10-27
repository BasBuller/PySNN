from collections import OrderedDict
import numpy as np
import torch


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
        self.defaults = defaults
        self.layers = layers

    def update_state(self):
        r"""Update state parameters of LearningRule based on latest network forward pass."""
        raise NotImplementedError

    def step(self):
        r"""Performs single learning step."""
        raise NotImplementedError

    def reset_state(self):
        r"""Reset state parameters of LearningRule."""
        raise NotImplementedError

    def check_layers(self, layers):
        r"""Check if layers provided to constructor are of the right format.
        
        :param layers: OrderedDict containing state dicts for each layer.
        """

        # Check if layers is iterator
        if not isinstance(layers, OrderedDict):
            raise TypeError(
                "Layers should be an iterator with deterministic ordering, a list, a tuple, or an OrderedDict. Current type is "
                + type(layers)
            )

        # Check for empty iterator
        if len(layers) == 0:
            raise ValueError("Got an empty layers iterator.")

        # Check for type of layers
        if not isinstance(list(layers.values())[0], (dict, OrderedDict)):
            raise TypeError(
                "A layer object should be a dict. Currently got a " + type(layers[0])
            )

    def pre_mult_post(self, pre, post, con_type):
        r"""Multiply a presynaptic term with a postsynaptic term, in the following order: pre x post.

        The outcome of this operation preserves batch size, but furthermore is directly broadcastable 
        with the weight of the connection.

        This operation differs for Linear or Convolutional connections. 

        :param pre: Presynaptic term
        :param post: Postsynaptic term
        :param con_type: Connection type, supports Linear and Conv2d

        :return: Tensor broadcastable with the weight of the connection
        """
        # Select target datatype
        if pre.dtype == torch.bool:
            pre = pre.to(post.dtype)
        elif post.dtype == torch.bool:
            post = post.to(pre.dtype)
        elif pre.dtype != post.dtype:
            assert TypeError(
                "The pre and post synaptic terms should either be of the same datatype, or one of them has to be a Boolean."
            )

        # Perform actual multiplication
        if con_type == "linear":
            pre = pre.transpose(2, 1)
        elif con_type == "conv2d":
            pre = pre.transpose(2, 1)
            post = post.view(post.shape[0], 1, post.shape[1], -1)

        output = pre * post
        return output.transpose(2, 1)

    def reduce_connections(self, tensor, con_type, red_method=torch.mean):
        r"""Reduces the tensor along the dimensions that represent seperate connections to an element of the weight Tensor.

        The function used for reducing has to be a callable that can be applied to single axes of a tensor.
        
        This operation differs or Linear or Convolutional connections.
        For Linear, only the batch dimension (dim 0) is reduced.
        For Conv2d, the batch (dim 0) and the number of kernel multiplications dimension (dim 3) are reduced.

        :param tensor: Tensor that will be reduced
        :param con_type: Connection type, support Linear and Conv2d
        :param red_method: Method used to reduce each dimension

        :return: Reduced Tensor
        """
        if con_type == "linear":
            output = red_method(tensor, dim=0)
        if con_type == "conv2d":
            output = red_method(tensor, dim=(0, 3))
        return output


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
