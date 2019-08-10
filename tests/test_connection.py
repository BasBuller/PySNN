import pytest
import torch


# NOTE: Base class is not tested, propagate spike is tested in the linear layer due to initialization problems of the weights


##########################################################
# Test delayed propagation of spikes
##########################################################
@pytest.fixture(scope="function", params=[
    # in_features, out_features, batch_size, dt, delay, tau_t, alpha_t 
    (2,            4,            1,          1,  1,     1.,    1.),
])
def delayed_connection(request):
    from pysnn.connection import LinearExponential

    params = request.param
    connection = LinearExponential(*params)
    connection.init_connection()
    return connection


# Test spike propagation
@pytest.mark.parametrize("spikes_in,spikes_out", [
    (torch.zeros(1, 1, 2, dtype=torch.float), torch.zeros(1, 4, 2, dtype=torch.float)),
    (torch.ones(1, 1, 2, dtype=torch.float), torch.ones(1, 4, 2, dtype=torch.float)),
    (torch.tensor([[[1, 0]]], dtype=torch.float), torch.tensor([[[1, 0], [1, 0], [1, 0], [1, 0]]], dtype=torch.float))
])
def test_propagate_delayed(spikes_in, spikes_out, delayed_connection):
    r"""Same function as for conv layer, so only tested here."""
    conn_out = delayed_connection.propagate_spike(spikes_in)
    empty_out = torch.zeros_like(spikes_out)
    assert (conn_out == empty_out).all()

    # Loop over delay until output from connection
    empty_in = torch.zeros_like(spikes_in)
    n_timesteps = delayed_connection.delay_init.view(-1)[0].int() - 1
    for delay_step in range(n_timesteps):
        conn_out = delayed_connection.propagate_spike(empty_in)
    assert (conn_out == spikes_out).all()

    # Make sure no secondary spikes occur
    conn_out = delayed_connection.propagate_spike(empty_in)
    assert (conn_out == empty_out).all()


##########################################################
# Test instant propagation of spikes
##########################################################
@pytest.fixture(scope="function", params=[
    # in_features, out_features, batch_size, dt, delay, tau_t, alpha_t 
    (2,            4,            1,          1,  0,     1.,    1.)
])
def instant_connection(request):
    from pysnn.connection import LinearExponential

    params = request.param
    connection = LinearExponential(*params)
    connection.init_connection()
    return connection


# Test spike propagation
@pytest.mark.parametrize("spikes_in,spikes_out", [
    (torch.zeros(1, 1, 2, dtype=torch.float), torch.zeros(1, 4, 2, dtype=torch.float)),
    (torch.ones(1, 1, 2, dtype=torch.float), torch.ones(1, 4, 2, dtype=torch.float)),
    (torch.tensor([[[1, 0]]], dtype=torch.float), torch.tensor([[[1, 0], [1, 0], [1, 0], [1, 0]]], dtype=torch.float))
])
def test_propagate_instant(spikes_in, spikes_out, instant_connection):
    r"""Same function as for conv layer, so only tested here."""
    conn_out = instant_connection.propagate_spike(spikes_in)
    assert (conn_out == spikes_out).all()


##########################################################
# Test linear layer functions
##########################################################
@pytest.fixture(scope="function", params=[
    # in_features, out_features, batch_size, dt, delay, tau_t, alpha_t 
    (2,            4,            1,          1,  1,     1.,    1.),
])
def linear_connection(request):
    from pysnn.connection import LinearExponential

    params = request.param
    connection = LinearExponential(*params)
    connection.init_connection()
    return connection


@pytest.mark.parametrize("spikes_in,trace_out", [
    (torch.ones(1, 1, 2, dtype=torch.uint8), torch.ones(1, 4, 2))
])
def test_linear_trace_update(spikes_in, trace_out, linear_connection):
    assert (linear_connection.trace == torch.zeros_like(trace_out)).all()
