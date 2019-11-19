import pytest
import torch


##########################################################
# General parameters, no reason to use a fixture due to simplicity of these tests
##########################################################
cells_shape = (1, 2, 2, 2)
threshold = 1.0
v_rest = 0.0
alpha_v = 1.0
alpha_t = 1.0
alpha_thresh = 0.9
dt = 1.0
tau_v = 2.0  # for exponential updates
tau_t = 2.0
tau_thresh = 2.0
v_decay = 0.5  # for linear updates
trace_decay = 0.5
thresh_decay = 0.5
refrac_counts = torch.zeros(*cells_shape)


##########################################################
# Test trace updates
##########################################################
@pytest.mark.parametrize(
    "trace, spikes_in, trace_out",
    [
        (
            torch.zeros(*cells_shape),
            torch.ones(*cells_shape),
            [torch.ones(*cells_shape) * 0.5, torch.ones(*cells_shape) * 0.75],
        )
    ],
)
def test_exponential_trace_update(trace, spikes_in, trace_out):
    from pysnn.functional import _exponential_trace_update

    for tr_out in trace_out:
        trace = _exponential_trace_update(trace, spikes_in, alpha_t, tau_t, dt)
        assert (trace == tr_out).all()


@pytest.mark.parametrize(
    "trace, spikes_in, trace_out",
    [
        (
            torch.zeros(*cells_shape),
            torch.ones(*cells_shape),
            [torch.ones(*cells_shape), torch.ones(*cells_shape) * 1.5],
        )
    ],
)
def test_linear_trace_update(trace, spikes_in, trace_out):
    from pysnn.functional import _linear_trace_update

    for tr_out in trace_out:
        trace = _linear_trace_update(trace, spikes_in, alpha_t, trace_decay, dt)
        assert (trace == tr_out).all()


##########################################################
# Test threshold updates for adaptive neurons
##########################################################
@pytest.mark.parametrize(
    "thresh, spikes_in, thresh_out",
    [
        (
            torch.ones(*cells_shape),
            torch.ones(*cells_shape),
            [torch.ones(*cells_shape) * 0.95, torch.ones(*cells_shape) * 0.925],
        )
    ],
)
def test_exponential_thresh_update(thresh, spikes_in, thresh_out):
    from pysnn.functional import _exponential_thresh_update

    for th_out in thresh_out:
        thresh = _exponential_thresh_update(
            thresh, spikes_in, alpha_thresh, tau_thresh, dt
        )
        assert thresh.allclose(th_out)


@pytest.mark.parametrize(
    "thresh, spikes_in, thresh_out",
    [
        (
            torch.ones(*cells_shape),
            torch.ones(*cells_shape),
            [torch.ones(*cells_shape) * 1.4, torch.ones(*cells_shape) * 1.6],
        )
    ],
)
def test_linear_thresh_update(thresh, spikes_in, thresh_out):
    from pysnn.functional import _linear_thresh_update

    for th_out in thresh_out:
        thresh = _linear_thresh_update(
            thresh, spikes_in, alpha_thresh, thresh_decay, dt
        )
        assert thresh.allclose(th_out)


##########################################################
# Test neuron voltage updates
##########################################################
# Test IF voltage update
@pytest.mark.parametrize(
    "v_cur, v_in, v_out",
    [
        (
            torch.zeros(*cells_shape),
            torch.ones(*cells_shape),
            [torch.ones(*cells_shape), torch.ones(*cells_shape) * 2.0],
        )
    ],
)
def test_if_voltage_update(v_cur, v_in, v_out):
    from pysnn.functional import _if_voltage_update

    for volt_out in v_out:
        v_cur = _if_voltage_update(v_cur, v_in, alpha_v, refrac_counts)
        assert (v_cur == volt_out).all()


# Test LIF linear voltage update
@pytest.mark.parametrize(
    "v_cur, v_in, v_out",
    [
        (
            torch.zeros(*cells_shape),
            torch.ones(*cells_shape) * 2.0,
            [torch.ones(*cells_shape) * 2.0, torch.ones(*cells_shape) * 3.0],
        )
    ],
)
def test_lif_linear_voltage_update(v_cur, v_in, v_out):
    from pysnn.functional import _lif_linear_voltage_update

    for volt_out in v_out:
        v_cur = _lif_linear_voltage_update(
            v_cur, v_rest, v_in, alpha_v, v_decay, dt, refrac_counts
        )
        assert (v_cur == volt_out).all()


# Test LIF exponential voltage update
@pytest.mark.parametrize(
    "v_cur, v_in, v_out",
    [
        (
            torch.zeros(*cells_shape),
            torch.ones(*cells_shape) * 2,
            [torch.ones(*cells_shape) * 1, torch.ones(*cells_shape) * 1.5],
        )
    ],
)
def test_lif_exponential_voltage_update(v_cur, v_in, v_out):
    from pysnn.functional import _lif_exponential_voltage_update

    for volt_out in v_out:
        v_cur = _lif_exponential_voltage_update(
            v_cur, v_rest, v_in, alpha_v, tau_v, dt, refrac_counts
        )
        assert (v_cur == volt_out).all()


# Test Fede voltage update
@pytest.mark.parametrize(
    "v_cur,v_in,trace_in,v_out",
    [
        (
            torch.zeros(*cells_shape),
            torch.ones(*cells_shape, 2) * 2.0,
            torch.ones(*cells_shape, 2),
            [torch.ones(*cells_shape), torch.ones(*cells_shape) * 1.5],
        )
    ],
)
def test_fede_voltage_update(v_cur, v_in, trace_in, v_out):
    from pysnn.functional import _fede_voltage_update

    for volt_out in v_out:
        v_cur = _fede_voltage_update(
            v_cur, v_rest, v_in, alpha_v, tau_v, dt, refrac_counts, trace_in
        )
        assert (v_cur == volt_out).all()
