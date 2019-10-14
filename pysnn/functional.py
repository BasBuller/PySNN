import torch


########################################################
# Trace updates, both neurons and layers
########################################################
def _exponential_trace_update(trace, x, alpha_t, tau_t, dt):
    r"""Calculate change in cell's trace based on current trace and incoming spikes x."""
    trace += (dt / tau_t) * (-trace + alpha_t * x)
    # TODO: Check for possible inplace instead of copying operation, should be inplace for best performance
    return trace


def _linear_trace_update(trace, x, alpha_t, trace_decay, dt):
    r"""Calculate change in cell's trace based on a fixed decay factor and incoming spikes x."""
    trace *= trace_decay
    trace += alpha_t * x
    # TODO: Check for possible inplace instead of copying operation, should be inplace for best performance
    return trace


########################################################
# Neuron threshold updates
########################################################
def _exponential_thresh_update(thresh, x, alpha_thresh, tau_thresh, dt):
    r"""Calculate change in cell's threshold based on current threshold and incoming spikes x."""
    thresh += (dt / tau_thresh) * (-thresh + alpha_thresh * x)
    # TODO: Check for possible inplace instead of copying operation, should be inplace for best performance
    return thresh


def _linear_thresh_update(thresh, x, alpha_thresh, thresh_decay, dt):
    r"""Calculate change in cell's threshold based on a fixed decay factor and incoming spikes x."""
    thresh *= thresh_decay
    thresh += alpha_thresh * x
    # TODO: Check for possible inplace instead of copying operation, should be inplace for best performance
    return thresh


########################################################
# Neuron voltage updates
########################################################
def _if_voltage_update(v_cur, v_in, alpha, refrac_counts):
    r"""Calculate change in cell's voltage based on current and incoming voltage."""
    v_delta = alpha * v_in
    non_refrac = refrac_counts == 0
    v_cur += v_delta * non_refrac.to(v_delta.dtype)
    # TODO: Check for possible inplace instead of copying operation, should be inplace for best performance
    return v_cur


def _lif_linear_voltage_update(
    v_cur, v_rest, v_in, alpha_v, v_decay, dt, refrac_counts
):
    r"""Calculate change in cell's voltage based on a linear relation between current and incoming voltage."""
    v_delta = (v_cur - v_rest) * v_decay + alpha_v * v_in
    non_refrac = refrac_counts == 0
    v_cur = v_rest + v_delta * non_refrac.to(v_delta.dtype)
    # TODO: Check for possible inplace instead of copying operation, should be inplace for best performance
    return v_cur


def _lif_exponential_voltage_update(
    v_cur, v_rest, v_in, alpha_v, tau_v, dt, refrac_counts
):
    r"""Calculate change in cell's voltage based on current and incoming voltage."""
    v_delta = (dt / tau_v) * (-(v_cur - v_rest) + alpha_v * v_in)
    non_refrac = refrac_counts == 0
    v_cur += v_delta * non_refrac.to(v_delta.dtype)
    # TODO: Check for possible inplace instead of copying operation, should be inplace for best performance
    return v_cur


def _fede_voltage_update(
    v_cur, v_rest, v_in, alpha_v, tau_v, dt, refrac_counts, pre_trace
):
    r"""Calculate change in cell's voltage based on current voltage and input trace."""
    forcing = alpha_v * (v_in - pre_trace).sum(-1)
    v_delta = (dt / tau_v) * (-(v_cur - v_rest) + forcing)
    non_refrac = refrac_counts == 0
    v_cur += v_delta * non_refrac.to(v_delta.dtype)
    # TODO: Check for possible inplace instead of copying operation, should be inplace for best performance
    return v_cur
