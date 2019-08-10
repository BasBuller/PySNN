import pytest
import torch


##########################################################
# General parameters, no reason to use a fixture due to simplicity of these tests
##########################################################
cells_shape = (1, 2, 2, 2)
thresh = 1
v_rest = 0
alpha_v = 1
alpha_t = 1
dt = 1
duration_refrac = 1
tau_v = 2
tau_t = 2
refrac_counts = torch.zeros(*cells_shape)


##########################################################
# Test general functionals
##########################################################
@pytest.mark.parametrize("trace,spikes_in,trace_out", [
    (torch.zeros(*cells_shape), torch.ones(*cells_shape), 
        [torch.ones(*cells_shape)*0.5, torch.ones(*cells_shape)*0.75])
])
def test_exponential_trace_update(trace, spikes_in, trace_out):
    from pysnn.functional import _exponential_trace_update

    for tr_out in trace_out:
        trace += _exponential_trace_update(trace, spikes_in, alpha_t, tau_t, dt)
        assert (trace == tr_out).all()


##########################################################
# Test neuron functionals
##########################################################
# Test IF voltage update
@pytest.mark.parametrize("v_cur,v_in,v_out", [
    (torch.zeros(*cells_shape), torch.ones(*cells_shape), [torch.ones(*cells_shape), torch.ones(*cells_shape)])
])
def test_lif_voltage_update(v_cur, v_in, v_out):
    from pysnn.functional import _if_voltage_update

    for volt_out in v_out:
        v_cur = _if_voltage_update(v_cur, v_in, alpha_v, refrac_counts)
        assert (v_cur == volt_out).all()


# Test LIF voltage update
@pytest.mark.parametrize("v_cur,v_in,v_out", [
    (torch.zeros(*cells_shape), torch.ones(*cells_shape)*2, 
        [torch.ones(*cells_shape)*1, torch.ones(*cells_shape)*1.5])
])
def test_lif_voltage_update(v_cur, v_in, v_out):
    from pysnn.functional import _lif_voltage_update

    for volt_out in v_out:
        v_cur = _lif_voltage_update(v_cur, v_rest, v_in, alpha_v, tau_v, dt, refrac_counts)
        assert (v_cur == volt_out).all()


# Test fede voltage update
@pytest.mark.parametrize("v_cur,v_in,trace_in,v_out", [
    (torch.zeros(*cells_shape), torch.ones(*cells_shape)*3, torch.ones(2, *cells_shape), 
        [torch.ones(*cells_shape)*0.5, torch.ones(*cells_shape)*0.75])
])
def test_fede_voltage_update(v_cur, v_in, trace_in, v_out):
    from pysnn.functional import _fede_voltage_update

    for volt_out in v_out:
        v_cur = _fede_voltage_update(v_cur, v_rest, v_in, alpha_v, tau_v, dt, refrac_counts, trace_in)
        assert (v_cur == volt_out).all()
