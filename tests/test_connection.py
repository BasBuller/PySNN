import pytest
import torch


# NOTE: Base class is not tested, propagate spike is tested in the linear layer due to initialization problems of the weights


##########################################################
# Global parameters
##########################################################


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
lnr_in_feat = 5
lnr_out_feat = 10
lnr_batch = 2
lnr_inpt_shape = (lnr_batch, 1, lnr_in_feat)
lnr_outpt_shape = (lnr_batch, 1, lnr_out_feat, lnr_in_feat)
lnr_wgt_shape = (lnr_out_feat, lnr_in_feat)

@pytest.fixture(scope="function", params=[
    # in_features, out_features, batch_size, dt, delay, tau_t, alpha_t, weight 
    (lnr_in_feat,  lnr_out_feat, lnr_batch,  1,  0,     1.,    1.,      0.5),
])
def linear_connection(request):
    from pysnn.connection import LinearExponential

    params = request.param
    connection = LinearExponential(*params[:-1])
    connection.init_connection()
    connection.weight.fill_(params[-1])
    return connection


@pytest.mark.parametrize("spikes_in,weights,potential_out", [
    (torch.ones(*lnr_outpt_shape), torch.ones(*lnr_wgt_shape)*0.5, torch.ones(*lnr_outpt_shape)*0.5),
    (torch.ones(*lnr_outpt_shape), torch.ones(*lnr_wgt_shape)*1, torch.ones(*lnr_outpt_shape)*1),
    (torch.ones(*lnr_outpt_shape), torch.ones(*lnr_wgt_shape)*2, torch.ones(*lnr_outpt_shape)*2),
    (torch.ones(*lnr_outpt_shape), torch.ones(*lnr_wgt_shape)*3, torch.ones(*lnr_outpt_shape)*3)
])
def test_linear_activation_potential(spikes_in, weights, potential_out, linear_connection):
    linear_connection.weight.data = weights
    potential = linear_connection.activation_potential(spikes_in)
    assert (potential == potential_out).all()


@pytest.mark.parametrize("spikes_in,activation_out,trace_out", [
    (torch.ones(*lnr_inpt_shape, dtype=torch.uint8), torch.ones(*lnr_outpt_shape)*0.5, torch.ones(*lnr_outpt_shape)),
])
def test_linear_forward(spikes_in, activation_out, trace_out, linear_connection):
    activation, trace = linear_connection.forward(spikes_in)
    assert (activation == activation_out).all()
    assert (trace == trace_out).all()


##########################################################
# Test Conv2d layer functions
##########################################################
conv_batch = 2
conv_chan_in = 1
conv_chan_out = 2
conv_in_im = (10, 10)
conv_out_im = (8, 8)
conv_kernel = (3, 3)

conv_inpt_shape = (conv_batch, conv_chan_in, *conv_in_im)
conv_outpt_shape = (conv_batch, conv_chan_in, *conv_out_im, 3*3)
conv_wgt_shape = (conv_chan_out, conv_chan_in, *conv_kernel)

@pytest.fixture(scope="function", params=[
    (conv_chan_in, conv_chan_out, conv_kernel, conv_in_im, conv_batch, 1, 0, 1, 1, 0.5)  # Last item is weight init
])
def conv2d_connection(request):
    from pysnn.connection import Conv2dExponential

    params = request.param
    connection = Conv2dExponential(*params[:-1])
    connection.init_connection()
    connection.weight.fill_(params[-1])
    return connection


@pytest.mark.parametrize("spikes_in,weights,potential_out", [
    (torch.ones(conv_batch, conv_chan_in, 3*3, 8*8), torch.ones(*conv_wgt_shape)*0.5, torch.ones(*conv_outpt_shape)*0.5)
])
def test_conv2d_activation_potential(spikes_in, weights, potential_out, conv2d_connection):
    conv2d_connection.weight.data = weights
    potential = conv2d_connection.activation_potential(spikes_in)
    assert (potential == potential_out).all()


@pytest.mark.parametrize("spikes_in,activation_out,trace_out", [
    (torch.ones(*conv_inpt_shape, dtype=torch.uint8), torch.ones(*conv_outpt_shape)*0.5, torch.ones(*conv_outpt_shape)),
])
def test_conv2d_forward(spikes_in, activation_out, trace_out, conv2d_connection):
    activation, trace = conv2d_connection.forward(spikes_in)
    assert (activation == activation_out).all()
    assert (trace == trace_out).all()
