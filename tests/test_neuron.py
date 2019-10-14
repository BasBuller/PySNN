import pytest
import torch


##########################################################
# Test Base Neuron
##########################################################
@pytest.fixture(
    scope="function",
    params=[
        # cell_shape, thresh, v_rest, alpha_v, alpha_t, dt, duration_refrac
        ((2, 2), 0.5, 0.0, 0.3, 1.0, 1.0, 2)
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
    "mask,voltage,spikes",
    [
        (torch.ones(2, 2, dtype=torch.uint8), 1, 4),
        (torch.ones(2, 2, dtype=torch.uint8), 0.5, 4),
        (torch.ones(2, 2, dtype=torch.uint8), 0, 0),
        (torch.tensor([[1, 0], [0, 1]], dtype=torch.uint8), 1, 2),
        (torch.tensor([[1, 0], [0, 0]], dtype=torch.uint8), 1, 1),
    ],
)
def test_spiking(mask, voltage, spikes, neuron):
    r"""Checks if the correct neurons spike based on neuron voltage."""
    neuron.v_cell.masked_fill_(mask.bool(), voltage)
    assert neuron.spiking().sum() == spikes


# Test refrac
@pytest.mark.parametrize(
    "spikes",
    [
        torch.tensor([[1, 1], [1, 1]], dtype=torch.uint8),
        torch.tensor([[1, 0], [0, 1]], dtype=torch.uint8),
    ],
)
def test_refrac(spikes, neuron):
    r"""Check if refrac counting is correct"""
    assert (neuron.refrac_counts == 0).all(), "Refrac count not initiated at zero"

    # Setup voltage for check reset
    neuron.v_cell.fill_(1)
    assert (neuron.v_cell > 0).all()

    total_spiking = spikes.sum()
    neuron.refrac(spikes.bool())
    assert (
        neuron.v_cell[spikes.bool()] == 0
    ).all()  # Check voltage reset on spiking neurons

    # Run counter down to zero
    zero_spikes = torch.zeros_like(spikes)
    for _ in range(neuron.duration_refrac.to(torch.uint8) - 1):
        neuron.refrac(zero_spikes.bool())
        neurons_in_refrac = (neuron.refrac_counts > 0).sum()
        assert neurons_in_refrac == total_spiking

    # Last increment, check all cells out of refrac state
    neuron.refrac(zero_spikes.bool())
    neurons_in_refrac = (neuron.refrac_counts > 0).sum()
    assert neurons_in_refrac == 0


##########################################################
# Test IF neuron
##########################################################
@pytest.fixture(
    scope="function",
    params=[
        # cells_shape, thresh, v_rest, alpha_v, alpha_t, dt, duration_refrac, tau_t
        ((1, 2, 2, 2), 1.0, 0.0, 1.0, 1.0, 1.0, 1, 0.9)
    ],
)
def if_neuron(request):
    from pysnn.neuron import IFNeuron

    params = request.param
    neuron = IFNeuron(*params)
    neuron.init_neuron()
    return neuron


# Test forward
@pytest.mark.parametrize(
    "spikes,spikes_out",
    [[torch.ones(1, 2, 2, 2) * 4, torch.ones(1, 2, 2, 2, dtype=torch.uint8)]],
)
def test_lif_forward(spikes, spikes_out, if_neuron):
    # Test correct output spiking pattern
    cell_out, trace_out = if_neuron.forward(spikes)
    assert (cell_out.byte() == spikes_out).all()
    assert (trace_out == spikes_out.float() * if_neuron.alpha_t).all()


##########################################################
# Test LIF neuron
##########################################################
@pytest.fixture(
    scope="function",
    params=[
        # cells_shape, thresh, v_rest, alpha_v, alpha_t, dt, duration_refrac, tau_v, tau_t
        ((1, 2, 2, 2), 1.0, 0.0, 1.0, 1.0, 1.0, 1, 0.9, 0.9)
    ],
)
def lif_neuron(request):
    from pysnn.neuron import LIFNeuron

    params = request.param
    neuron = LIFNeuron(*params)
    neuron.init_neuron()
    return neuron


# Test forward
@pytest.mark.parametrize(
    "spikes,spikes_out",
    [[torch.ones(1, 2, 2, 2) * 4, torch.ones(1, 2, 2, 2, dtype=torch.uint8)]],
)
def test_lif_forward(spikes, spikes_out, lif_neuron):
    # Test correct output spiking pattern
    cell_out, trace_out = lif_neuron.forward(spikes)
    assert (cell_out.byte() == spikes_out).all()
    assert (trace_out == spikes_out.float() * lif_neuron.alpha_t).all()


##########################################################
# Test adaptive LIF neuron
##########################################################
@pytest.fixture(
    scope="function",
    params=[
        # cells_shape, thresh, v_rest, alpha_v, alpha_t, dt, duration_refrac, tau_v, tau_t, alpha_thresh, tau_thresh
        ((1, 2, 2, 2), 1.0, 0.0, 1.0, 1.0, 1.0, 1, 0.9, 0.9, 1.0, 0.9)
    ],
)
def adaptive_lif_neuron(request):
    from pysnn.neuron import AdaptiveLIFNeuron

    params = request.param
    neuron = AdaptiveLIFNeuron(*params)
    neuron.init_neuron()
    return neuron


# Test forward
@pytest.mark.parametrize(
    "spikes,spikes_out",
    [[torch.ones(1, 2, 2, 2) * 4, torch.ones(1, 2, 2, 2, dtype=torch.uint8)]],
)
def test_adaptive_lif_forward(spikes, spikes_out, adaptive_lif_neuron):
    # Test correct output spiking pattern
    cell_out, trace_out = adaptive_lif_neuron.forward(spikes)
    assert (cell_out.byte() == spikes_out).all()
    assert (trace_out == spikes_out.float() * adaptive_lif_neuron.alpha_t).all()
    print(adaptive_lif_neuron.thresh)
    print(adaptive_lif_neuron.thresh_center)
    print(adaptive_lif_neuron.alpha_thresh)
    print(spikes_out)
    assert (
        adaptive_lif_neuron.thresh
        == adaptive_lif_neuron.thresh_center * adaptive_lif_neuron.tau_thresh
        + spikes_out.float() * adaptive_lif_neuron.alpha_thresh
    ).all()


##########################################################
# Test Fede's neuron
##########################################################
@pytest.fixture(
    scope="function",
    params=[
        # cells_shape, thresh, v_rest, alpha_v, alpha_t, dt, duration_refrac, tau_v, tau_t
        ((1, 2, 2, 2), 1.0, 0.0, 1.0, 1.0, 1.0, 1, 2.0, 2.0)
    ],
)
def fede_neuron(request):
    from pysnn.neuron import FedeNeuron

    params = request.param
    neuron = FedeNeuron(*params)
    neuron.init_neuron()
    return neuron


# Test forward
@pytest.mark.parametrize(
    "spikes,trace_in,spikes_out",
    [
        [
            torch.ones(1, 2, 2, 2) * 4,
            torch.ones(1, 2, 2, 2, 2),
            torch.ones(1, 2, 2, 2, dtype=torch.uint8),
        ]
    ],
)
def test_fede_forward(spikes, trace_in, spikes_out, fede_neuron):
    # Test correct output spiking pattern
    cell_out, trace_out = fede_neuron.forward(spikes, trace_in)
    assert (cell_out.byte() == spikes_out).all()
