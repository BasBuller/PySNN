import pytest as test

from pysnn.neuron import Neuron, IFNeuronTrace, LIFNeuronTrace, FedeNeuronTrace


##########################################################
# Testing Neuron
##########################################################
cells_shape = []
thresh = 0.5
v_rest = 0.
alpha_v = 0.3
alpha_t = 1.
dt = 1.
duration_refrac = 5.

def test_neuron_refrac():
    neuron = Neuron(cells_shape, thresh, v_rest, alpha_v, alpha_t, dt, duration_refrac)
    assert 0
