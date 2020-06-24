import pytest
from collections import OrderedDict
import torch


##########################################################
# Test LearningRule
##########################################################
@pytest.fixture(
    scope="function",
    params=[
        # layers, defaults
        (OrderedDict(), {})
    ],
)
def learning_rule(request):
    from pysnn.learning import LearningRule

    params = request.param
    rule = LearningRule(*params)
    return rule


# Test pre-post multiplication
@pytest.mark.parametrize(
    "pre, post, conn, result",
    [
        (
            torch.tensor([[[0, 1, 1, 0]], [[1, 0, 0, 1]]], dtype=torch.bool),
            torch.tensor([[[0.0, 1.0]], [[1.0, 0.0]]], dtype=torch.float),
            "linear",
            torch.tensor(
                [
                    [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]],
                    [[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
                ],
                dtype=torch.float,
            ),
        ),
        (
            torch.ones(2, 4, 50, 1024, dtype=torch.bool),
            torch.ones(2, 4, 32, 32, dtype=torch.float),
            "conv2d",
            torch.ones(2, 4, 50, 1024),
        ),
    ],
)
def test_pre_mult_post(pre, post, conn, result, learning_rule):
    r"""Checks the multiplication of spikes/traces of pre and post neurons."""
    from pysnn.connection import Linear, Conv2d

    if conn == "linear":
        # in, out, batch, dt, delay
        conn_params = (1, 1, 1, 1.0, 0)
        conn = Linear(*conn_params)
    elif conn == "conv2d":
        # in, out, kernel, image, batch, dt, delay
        conn_params = (1, 1, 1, (1, 1), 1, 1.0, 0)
        conn = Conv2d(*conn_params)

    out = learning_rule.pre_mult_post(pre, post, conn)
    assert (out == result).all()


# Test reduce
@pytest.mark.parametrize(
    "tensor, conn, result",
    [
        (
            torch.tensor(
                [[[0.0, 1.0, 1.0, 0.0]], [[1.0, 0.0, 0.0, 1.0]]], dtype=torch.float
            ),
            "linear",
            torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float),
        ),
        (torch.ones(2, 4, 50, 1024, dtype=torch.float), "conv2d", torch.ones(4, 50)),
    ],
)
def test_reduce_connections(tensor, conn, result, learning_rule):
    r"""Checks the reduction to dimensions that represent separate connections."""
    from pysnn.connection import Linear, Conv2d

    if conn == "linear":
        # in, out, batch, dt, delay
        conn_params = (1, 1, 1, 1.0, 0)
        conn = Linear(*conn_params)
    elif conn == "conv2d":
        # in, out, kernel, image, batch, dt, delay
        conn_params = (1, 1, 1, (1, 1), 1, 1.0, 0)
        conn = Conv2d(*conn_params)

    out = learning_rule.reduce_connections(tensor, conn)
    assert (out == result).all()


##########################################################
# Create layers for learning rule testing
##########################################################
@pytest.fixture(
    scope="function",
    params=[
        # con_type, pre_shape, post_shape, n_dynamics, c_dynamics
        (
            "linear",
            (2, 1, 6),
            (2, 1, 10),
            (1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5),
            (6, 10, 2, 1.0, 0),
        ),
        (
            "conv2d",
            (2, 2, 10, 10),
            (2, 4, 8, 8),
            (1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5),
            (2, 4, 5, (10, 10), 2, 1.0, 0),
        ),
    ],
)
def layer(request):
    from pysnn.network import SNNNetwork
    from pysnn.neuron import LIFNeuron
    from pysnn.connection import Linear, Conv2d

    params = request.param
    pre_neuron = LIFNeuron(params[1], *params[3])
    post_neuron = LIFNeuron(params[2], *params[3])
    if params[0] == "linear":
        conn = Linear(*params[4])
    elif params[0] == "conv2d":
        conn = Conv2d(*params[4], padding=1)

    network = SNNNetwork()
    network.add_layer("layer", conn, post_neuron)
    return network.layer_state_dict()


##########################################################
# Test MSTDPET
##########################################################
@pytest.fixture(
    scope="function",
    params=[
        # a_pre, a_post, lr, e_trace_decay
        (1.0, 1.0, 0.01, 0.9)
    ],
)
def mstdpet(request, layer):
    from pysnn.learning import MSTDPET

    params = request.param
    rule = MSTDPET(layer, *params)
    return rule


# Test MSTDPET step
@pytest.mark.parametrize("reward", [1.0])
def test_mstdpet_step(reward, mstdpet):
    r"""Checks whether a step with the MSTDPET learning rule works."""
    mstdpet.update_state()
    mstdpet.step(reward)


##########################################################
# Test FedeSTDP
##########################################################
@pytest.fixture(
    scope="function",
    params=[
        # lr, w_init, a
        (0.01, 0.5, 0.9)
    ],
)
def fede_stdp(request, layer):
    from pysnn.learning import FedeSTDP

    params = request.param
    rule = FedeSTDP(layer, *params)
    return rule


# Test FedeSTDP step
@pytest.fixture(scope="function", params=[])
def test_mstdpet_step(reward, fede_stdp):
    r"""Checks whether a step with the FedeSTDP learning rule works."""
    fede_stdp.step(reward)
