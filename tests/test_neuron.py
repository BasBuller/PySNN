import pytest
import torch


##########################################################
# Test Input Neuron
##########################################################
@pytest.fixture(
    scope="function",
    params=[
        # cell_shape, dt, alpha_t, tau_t
        ((1, 2, 2, 2), 1.0, 1.0, 0.5)
    ],
)
def input_neuron(request):
    from pysnn.neuron import Input

    params = request.param
    neuron = Input(*params)
    return neuron


# Test forward
@pytest.mark.parametrize(
    "inputs, expected",
    [
        (
            torch.ones(1, 2, 2, 2, dtype=torch.float) * 2,
            torch.ones(1, 2, 2, 2, dtype=torch.bool),
        )
    ],
)
def test_input_forward(inputs, expected, input_neuron):
    r"""Checks if inputs are copied correctly."""
    outputs, trace = input_neuron.forward(inputs)
    assert (inputs == outputs).all()
    assert (outputs.bool() == expected).all()
    assert (outputs.bool() == input_neuron.spikes).all()
    assert (trace == inputs * input_neuron.alpha_t).all()


##########################################################
# Test Base Neuron
##########################################################
@pytest.fixture(
    scope="function",
    params=[
        # cell_shape, thresh, v_rest, dt, duration_refrac
        ((1, 2, 2), 1.0, 0.0, 1.0, 3.0)
    ],
)
def neuron(request):
    from pysnn.neuron import BaseNeuron

    params = request.param
    neuron = BaseNeuron(*params)
    neuron.init_neuron()
    return neuron


# Test spiking
@pytest.mark.parametrize(
    "mask, voltage, spikes",
    [
        (torch.ones(1, 2, 2, dtype=torch.bool), 2.0, 4),
        (torch.ones(1, 2, 2, dtype=torch.bool), 1.0, 4),
        (torch.ones(1, 2, 2, dtype=torch.bool), 0.0, 0),
        (torch.tensor([[[1, 0], [0, 1]]], dtype=torch.bool), 2.0, 2),
        (torch.tensor([[[1, 0], [0, 0]]], dtype=torch.bool), 2.0, 1),
    ],
)
def test_spiking(mask, voltage, spikes, neuron):
    r"""Checks if the correct neurons spike based on neuron voltage."""
    neuron.v_cell.masked_fill_(mask, voltage)
    assert neuron.spiking().sum() == spikes


# Test refrac
@pytest.mark.parametrize(
    "spikes",
    [
        torch.tensor([[[1, 1], [1, 1]]], dtype=torch.bool),
        torch.tensor([[[1, 0], [0, 1]]], dtype=torch.bool),
    ],
)
def test_refrac(spikes, neuron):
    r"""Checks if refrac counting is correct"""
    assert (neuron.refrac_counts == 0).all(), "Refraction count not initiated at zero."

    # Setup voltage for check reset
    neuron.v_cell.fill_(1)
    assert (neuron.v_cell > 0).all()

    total_spiking = spikes.sum()
    neuron.refrac(spikes)
    assert (neuron.v_cell[spikes] == 0).all()  # Check voltage reset on spiking neurons

    # Run counter down to zero
    zero_spikes = torch.zeros_like(spikes)
    for _ in range(neuron.duration_refrac.byte() - 1):
        neuron.refrac(zero_spikes)
        neurons_in_refrac = (neuron.refrac_counts > 0).sum()
        assert neurons_in_refrac == total_spiking

    # Last increment, check all cells out of refraction state
    neuron.refrac(zero_spikes)
    neurons_in_refrac = (neuron.refrac_counts > 0).sum()
    assert neurons_in_refrac == 0


##########################################################
# Test IF Neuron
##########################################################
@pytest.fixture(
    scope="function",
    params=[
        # cells_shape, thresh, v_rest, alpha_v, alpha_t, dt, duration_refrac, tau_t
        ((1, 2, 2, 2), 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5)
    ],
)
def if_neuron(request):
    from pysnn.neuron import IFNeuron

    params = request.param
    neuron = IFNeuron(*params)
    return neuron


# Test forward
@pytest.mark.parametrize(
    "inputs, expected",
    [
        (
            torch.ones(1, 2, 2, 2, dtype=torch.float) * 2,
            torch.ones(1, 2, 2, 2, dtype=torch.bool),
        )
    ],
)
def test_lif_forward(inputs, expected, if_neuron):
    r"""Checks correct output spiking pattern."""
    spikes, trace = if_neuron.forward(inputs)
    assert (spikes == expected).all()
    assert (trace == expected.float() * if_neuron.alpha_t).all()


##########################################################
# Test LIF Neuron
##########################################################
@pytest.fixture(
    scope="function",
    params=[
        # cells_shape, thresh, v_rest, alpha_v, alpha_t, dt, duration_refrac, tau_v, tau_t
        ((1, 2, 2, 2), 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5)
    ],
)
def lif_neuron(request):
    from pysnn.neuron import LIFNeuron

    params = request.param
    neuron = LIFNeuron(*params)
    return neuron


# Test forward
@pytest.mark.parametrize(
    "inputs, expected",
    [
        (
            torch.ones(1, 2, 2, 2, dtype=torch.float) * 2,
            torch.ones(1, 2, 2, 2, dtype=torch.bool),
        )
    ],
)
def test_lif_forward(inputs, expected, lif_neuron):
    r"""Checks correct output spiking pattern."""
    spikes, trace = lif_neuron.forward(inputs)
    assert (spikes == expected).all()
    assert (trace == expected.float() * lif_neuron.alpha_t).all()


##########################################################
# Test Adaptive LIF Neuron
##########################################################
@pytest.fixture(
    scope="function",
    params=[
        # cells_shape, thresh, v_rest, alpha_v, alpha_t, dt, duration_refrac, tau_v, tau_t, alpha_thresh, tau_thresh
        ((1, 2, 2, 2), 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5)
    ],
)
def adaptive_lif_neuron(request):
    from pysnn.neuron import AdaptiveLIFNeuron

    params = request.param
    neuron = AdaptiveLIFNeuron(*params)
    return neuron


# Test forward
@pytest.mark.parametrize(
    "inputs, expected",
    [
        (
            torch.ones(1, 2, 2, 2, dtype=torch.float) * 2,
            torch.ones(1, 2, 2, 2, dtype=torch.bool),
        )
    ],
)
def test_adaptive_lif_forward(inputs, expected, adaptive_lif_neuron):
    r"""Checks correct output spiking pattern and threshold increase."""
    spikes, trace = adaptive_lif_neuron.forward(inputs)
    assert (spikes == expected).all()
    assert (trace == expected.float() * adaptive_lif_neuron.alpha_t).all()
    assert (
        adaptive_lif_neuron.thresh
        == adaptive_lif_neuron.thresh_center * adaptive_lif_neuron.tau_thresh
        + expected.float() * adaptive_lif_neuron.alpha_thresh
    ).all()


##########################################################
# Test Fede's neuron
##########################################################
@pytest.fixture(
    scope="function",
    params=[
        # cells_shape, thresh, v_rest, alpha_v, alpha_t, dt, duration_refrac, tau_v, tau_t
        ((1, 2, 2, 2), 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0)
    ],
)
def fede_neuron(request):
    from pysnn.neuron import FedeNeuron

    params = request.param
    neuron = FedeNeuron(*params)
    return neuron


# Test forward
@pytest.mark.parametrize(
    "inputs, pre_trace, expected",
    [
        (
            torch.ones(1, 2, 2, 2, dtype=torch.float) * 2,
            torch.ones(1, 2, 2, 2, 1, dtype=torch.float),
            torch.ones(1, 2, 2, 2, dtype=torch.bool),
        )
    ],
)
def test_fede_forward(inputs, pre_trace, expected, fede_neuron):
    r"""Checks correct output spiking pattern and threshold increase."""
    spikes, trace = fede_neuron.forward(inputs, pre_trace)
    assert (spikes == expected).all()
