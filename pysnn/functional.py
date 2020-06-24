import torch


########################################################
# Trace updates
########################################################
def exponential_trace_update(trace, x, alpha_t, tau_t, dt):
    r"""Calculate change in cell's trace based on current trace and incoming spikes x.

    Trace updates are performed according to the following formula:
    
    .. math::

        trace_{t+1} = trace_{t} + \frac{(-trace_{t} + alpha * x_{t}) * dt}{tau}

    :param trace: Current trace
    :param x: Incoming/presynaptic spikes
    :param alpha_t: Scaling factor for spike influence on trace
    :param tau_t: Trace decay time constant
    :param dt: Duration of single timestep

    :return: New trace values
    """
    trace += (dt / tau_t) * (-trace + alpha_t * x)
    # TODO: Check for possible inplace instead of copying operation, should be inplace for best performance
    return trace


def linear_trace_update(trace, x, alpha_t, trace_decay, dt):
    r"""Calculate change in cell's trace based on a fixed decay factor and incoming spikes x.

    Trace updates are performed according to the following formula:

    .. math::

        trace_{t+1} = trace_{t} * decay + alpha * x_{t}

    :param trace: Current trace
    :param x: Incoming/presynaptic spikes
    :param alpha_t: Scaling factor for spike influence on trace
    :param trace_decay: Trace decay factor, should be < 1
    :param dt: Duration of single timestep

    :return: New trace values
    """
    trace *= trace_decay
    trace += alpha_t * x
    # TODO: Check for possible inplace instead of copying operation, should be inplace for best performance
    return trace


########################################################
# Neuron threshold updates
########################################################
def exponential_thresh_update(thresh, thresh_rest, x, alpha_thresh, tau_thresh, dt):
    r"""Calculate change in cell's threshold based on current threshold and incoming spikes x.

    Threshold updates are performed according to the following formula:
    
    .. math::

        thresh_{t+1} = thresh_{t} + \frac{(-thresh_{t} + alpha * x_{t}) * dt}{tau}

    :param thresh: Current threshold
    :param x: Incoming/presynaptic spikes
    :param alpha_thresh: Scaling factor for spike influence on threshold
    :param tau_thresh: Threshold decay time constant
    :param dt: Duration of single timestep

    :return: New threshold values
    """
    thresh += (dt / tau_thresh) * (-(thresh - thresh_rest) + alpha_thresh * x)
    # TODO: Check for possible inplace instead of copying operation, should be inplace for best performance
    return thresh


def linear_thresh_update(thresh, thresh_rest, x, alpha_thresh, thresh_decay, dt):
    r"""Calculate change in cell's threshold based on a fixed decay factor and incoming spikes x.

    Threshold updates are performed according to the following formula:

    .. math::

        thresh_{t+1} = thresh_{t} * decay + alpha * x_{t}
    
    :param thresh: Current threshold
    :param x: Incoming/presynaptic spikes
    :param alpha_thresh: Scaling factor for spike influence on threshold
    :param thresh_decay: Threshold decay factor, should be < 1
    :param dt: Duration of single timestep

    :return: New threshold values
    """
    thresh *= thresh_decay
    thresh += alpha_thresh * x
    # TODO: Check for possible inplace instead of copying operation, should be inplace for best performance
    return thresh


########################################################
# Neuron voltage updates
########################################################
def if_voltage_update(v_cur, v_in, alpha, refrac_counts):
    r"""Calculate change in cell's voltage based on current and incoming voltage, no decay.
    
    :param v_cur: Current voltages
    :param v_in: Incoming voltages
    :param alpha: Incoming voltage scaling factor
    :param refrac_counts: Number of timesteps left in refractory state for each :class:`Neuron`

    :return: New voltages
    """
    v_delta = alpha * v_in
    non_refrac = refrac_counts == 0
    v_cur += v_delta * non_refrac.to(v_delta.dtype)
    # TODO: Check for possible inplace instead of copying operation, should be inplace for best performance
    return v_cur


def lif_linear_voltage_update(v_cur, v_rest, v_in, alpha_v, v_decay, dt, refrac_counts):
    r"""Calculate change in cell's voltage based on a linear relation between current and incoming voltage, with decay.

    Voltage updates are performed according to the following formula:

    .. math::

        voltage_{t+1} = rest_{t} + (voltage_{t} - rest_{t}) * decay + alpha * incoming_{t}

    :param v_cur: Current voltages
    :param v_rest: Resting voltages
    :param v_in: Incoming voltages
    :param alpha_v: Incoming voltage scaling factor
    :param v_decay: Voltage decay factor, should be < 1
    :param dt: Duration of single timestep
    :param refrac_counts: Number of timesteps left in refractory state for each :class:`Neuron`

    :return: New voltages
    """
    v_delta = (v_cur - v_rest) * v_decay + alpha_v * v_in
    non_refrac = refrac_counts == 0
    v_cur = v_rest + v_delta * non_refrac.to(v_delta.dtype)
    # TODO: Check for possible inplace instead of copying operation, should be inplace for best performance
    return v_cur


def lif_exponential_voltage_update(
    v_cur, v_rest, v_in, alpha_v, tau_v, dt, refrac_counts
):
    r"""Calculate change in cell's voltage based on current and incoming voltage.

    Voltage updates are performed according to the following formula:
    
    .. math::

        voltage_{t+1} = voltage_{t} + \frac{(-(voltage_{t} - rest_{t}) + alpha * incoming_{t}) * dt}{tau}

    :param v_cur: Current voltages
    :param v_rest: Resting voltages
    :param v_in: Incoming voltages
    :param alpha_v: Incoming voltage scaling factor
    :param tau_v: Voltage decay time constant
    :param dt: Duration of single timestep
    :param refrac_counts: Number of timesteps left in refractory state for each :class:`Neuron`

    :return: New voltages
    """
    v_delta = (dt / tau_v) * (-(v_cur - v_rest) + alpha_v * v_in)
    non_refrac = refrac_counts == 0
    v_cur += v_delta * non_refrac.to(v_delta.dtype)
    # TODO: Check for possible inplace instead of copying operation, should be inplace for best performance
    return v_cur


def fede_voltage_update(
    v_cur, v_rest, v_in, alpha_v, tau_v, dt, refrac_counts, pre_trace
):
    r"""Calculate change in cell's voltage based on current voltage and input trace.

    Defined in "Unsupervised Learning of a Hierarchical Spiking Neural Network for Optical Flow Estimation: From Events to Global Motion Perception" - F. Paredes-Valles, et al.

    Voltage updates are performed according to the following formula:
    
    .. math::

        voltage_{t+1} = voltage_{t} + \frac{(-(voltage_{t} - rest_{t}) + alpha * (incoming_{t} - pretrace_{t})) * dt}{tau}

    :param v_cur: Current voltages
    :param v_rest: Resting voltages
    :param v_in: Incoming voltages
    :param alpha_v: Incoming voltage scaling factor
    :param tau_v: Voltage decay time constant
    :param dt: Duration of single timestep
    :param refrac_counts: Number of timesteps left in refractory state for each :class:`Neuron`
    :param pre_trace: Incoming trace

    :return: New voltages
    """
    forcing = alpha_v * (v_in - pre_trace).sum(-1)
    v_delta = (dt / tau_v) * (-(v_cur - v_rest) + forcing)
    non_refrac = refrac_counts == 0
    v_cur += v_delta * non_refrac.to(v_delta.dtype)
    # TODO: Check for possible inplace instead of copying operation, should be inplace for best performance
    return v_cur






def forward(self, x, force_spike=False):
    r"""
    :param x: Incoming/presynaptic spikes

    :return: Neuron output spikes and trace
    """
    x = self.fold(x)
    self.update_voltage(x)
    spikes = self.spiking(force_spike=force_spike)
    self.update_trace(spikes)
    self.refrac(spikes)
    if self.complete_trace is not None:
        self.concat_trace(spikes)
    return spikes, self.trace

def refrac(self, spikes):
    r"""Basic counting version of cell refractory period.

    Can be overwritten in case of the need of more refined functionality.
    """
    if (self.duration_refrac > 0).any():
        self.refrac_counts[self.refrac_counts > 0] -= self.dt
        self.refrac_counts += self.duration_refrac * self.convert_spikes(spikes)
    self.v_cell.masked_fill_(spikes, self.v_rest)


########################################################
# Autgrad functions
########################################################
class BellecSpiking(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, v, v_th):
        ctx.v = v
        ctx.v_th = v_th
        return v >= v_th

    @staticmethod
    def backward(ctx, grad_output):
        abs_diff = torch.abs((ctx.v - ctx.v_th) / ctx.v_th)
        grad_v = 0.3 * torch.max(0, 1 - abs_diff)
        return grad_v, None


class Refrac(torch.autograd.Function):

    @staticmethod
    def forward(ctx):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass


class LIFLinearVoltageUpdate(torch.autograd.Function):

    @staticmethod
    def forward(ctx, v_cur, v_rest, v_in, alpha_v, v_decay, dt, refrac_counts, trace):
        v_delta = (v_cur - v_rest) * v_decay + alpha_v * v_in
        non_refrac = refrac_counts == 0
        v_cur = v_rest + v_delta * non_refrac.to(v_delta.dtype)

        ctx.trace = trace

        return v_cur

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, ctx.trace, None, None, None, None, None


########################################################
# Apply autograd functions
########################################################
bellec_spiking = BellecSpiking.apply
