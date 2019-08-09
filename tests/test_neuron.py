import pytest
import torch


##########################################################
# Neuron definition
##########################################################
# TODO: Can possibly insert multiple neuron designs
@pytest.fixture(scope="function")
def neuron():
    from pysnn.neuron import Neuron

    cells_shape = (2, 2)
    thresh = 0.5
    v_rest = 0.
    alpha_v = 0.3
    alpha_t = 1.
    dt = 1.
    duration_refrac = 5.

    neuron = Neuron(cells_shape, thresh, v_rest, alpha_v, alpha_t, dt, duration_refrac)
    neuron.init_neuron()
    return neuron


##########################################################
# Test spiking 
##########################################################
@pytest.mark.parametrize("mask,voltage,spikes", [
    (torch.ones(2, 2, dtype=torch.uint8), 1, 4),
    (torch.ones(2, 2, dtype=torch.uint8), 0.5, 4),
    (torch.ones(2, 2, dtype=torch.uint8), 0, 0),
    (torch.tensor([[1, 0], [0, 1]], dtype=torch.uint8), 1, 2)
])
def test_spiking(mask, voltage, spikes, neuron):
    neuron.v_cell.masked_fill_(mask, voltage)
    assert neuron.spiking().sum() == spikes


##########################################################
# Test refrac
##########################################################
