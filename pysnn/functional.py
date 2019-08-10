import torch


########################################################
# General functionals
########################################################
def _exponential_trace_update(trace, x, alpha_t, tau_t, dt):
    r"""Calculate change in cell's trace based on current trace output spike x."""
    return ((-trace * dt) + (alpha_t * x)) / tau_t


########################################################
# Neuron functionals
########################################################
def _if_voltage_update(v_cur, v_in, alpha, refrac_counts):
    r"""Calculate change in cell's voltage based on current and incoming voltage."""
    v_delta = alpha * v_in
    non_refrac = refrac_counts == 0
    v_cur += v_delta * non_refrac.to(v_delta.dtype)
    return v_cur


def _lif_voltage_update(v_cur, v_rest, v_in, alpha_v, tau_v, dt, refrac_counts):
    r"""Calculate change in cell's voltage based on current and incoming voltage."""
    v_delta = (-(v_cur - v_rest) * dt + alpha_v * v_in) / tau_v
    non_refrac = refrac_counts == 0
    v_cur += v_delta * non_refrac.to(v_delta.dtype)
    return v_cur


def _fede_voltage_update(v_cur, v_rest, v_in, alpha_v, tau_v, dt, refrac_counts, pre_trace):
    r"""Calculate change in cell's voltage based on current voltage and input trace."""
    forcing = alpha_v * (v_in - pre_trace.sum(0))
    v_delta = (-(v_cur - v_rest) * dt + forcing) / tau_v
    non_refrac = refrac_counts == 0
    v_cur += v_delta * non_refrac.to(v_delta.dtype)
    return v_cur

def _neuron_exponential_trace_update(trace, x, alpha_t, tau_t, dt):
    trace += _exponential_trace_update(trace, x, alpha_t, tau_t, dt).sum(1, keepdim=True)  # Sum over channel dimension! Both for conv and linear
    return trace


########################################################
# Connection functionals
########################################################
def _connection_exponential_trace_update(trace, x, alpha_t, tau_t, dt):
    trace += _exponential_trace_update(trace, x, alpha_t, tau_t, dt)
    return trace
