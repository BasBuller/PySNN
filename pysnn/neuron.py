import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from pysnn.utils import _set_no_grad
import pysnn.functional as sf


######################################################### 
# Base Neuron
#########################################################
class Neuron(nn.Module):
    r"""Base neuron model, is a container to define basic neuron functionalties.

    Defines basic spiking, voltage and trace characteristics. Just has to
    adhere to the API functionalities to integrate within Connection modules.

    Make sure the Neuron class receives input voltage for each neuron and
    returns a Tensor indicating which neurons have spiked.
    """
    def __init__(self,
                 cells_shape,
                 thresh,
                 v_rest,
                 alpha_v,
                 alpha_t,
                 dt,
                 duration_refrac):
        super(Neuron, self).__init__()
        self.cells_shape = torch.tensor(cells_shape)

        # Check compatibility of dt and refrac counting
        assert duration_refrac % dt == 0, "dt does not fit an integer amount of times in duration_refrac"

        # Fixed parameters
        self.v_rest = Parameter(torch.tensor(v_rest, dtype=torch.float))
        self.alpha_v = Parameter(torch.tensor(alpha_v, dtype=torch.float))  # TODO: Might want to move this out of base class
        self.alpha_t = Parameter(torch.tensor(alpha_t, dtype=torch.float))  # TODO: Might want to move this out of base class
        self.dt = Parameter(torch.tensor(dt, dtype=torch.float))
        self.duration_refrac = Parameter(torch.tensor(duration_refrac + 1, dtype=torch.float))
        self.thresh_center = Parameter(torch.tensor(thresh, dtype=torch.float))

        # Define dynamic parameters
        self.v_cell = Parameter(torch.Tensor(*cells_shape))
        self.trace = Parameter(torch.Tensor(*cells_shape))
        self.refrac_counts = Parameter(torch.Tensor(*cells_shape))

        # Define learnable parameters
        self.thresh = Parameter(torch.Tensor(*cells_shape))

    def spiking(self):
        r"""Return cells that are in spiking state."""
        return self.v_cell >= self.thresh

    def refrac(self, spikes):
        r"""Basic counting version of cell refractory period.

        Can be overwritten in case of the need of more refined functionality.
        """
        self.refrac_counts[self.refrac_counts > 0] -= self.dt
        self.refrac_counts += self.duration_refrac * self.convert_spikes(spikes)
        self.v_cell.masked_fill_(spikes, 0)

    def convert_spikes(self, spikes):
        return spikes.to(self.v_cell.dtype)

    def reset_state(self):
        r"""Reset cell states that accumulate over time during simulation."""
        self.v_cell.fill_(self.v_rest)
        self.refrac_counts.fill_(0)
        self.trace.fill_(0)

    def reset_parameters(self):
        r"""Reset learnable cell parameters to initialization values."""
        self.thresh.copy_(torch.ones_like(self.thresh.data) * self.thresh_center)

    def no_grad(self):
        r"""Turn off learning and gradient storing."""
        _set_no_grad(self)

    def init_neuron(self):
        r"""Initialize state, parameters and turn off gradients."""
        self.no_grad()
        self.reset_state()
        self.reset_parameters()

    def forward(self):
        return


#########################################################
# IF Neuron
#########################################################
class IFNeuronTrace(Neuron):
    r"""Integrate and Fire neuron."""
    def __init__(self,
                 cells_shape,
                 thresh,
                 v_rest,
                 alpha_v,
                 alpha_t,
                 dt,
                 duration_refrac,
                 tau_t):
        super(IFNeuronTrace, self).__init__(cells_shape, thresh, v_rest, alpha_v, alpha_t, dt, duration_refrac)

        #Fixed parameters
        self.tau_t = Parameter(torch.tensor(tau_t, dtype=torch.float))
        self.init_neuron()

    def update_trace(self, x):
        self.trace = sf._neuron_exponential_trace_update(self.trace, x, self.alpha_t, self.tau_t, self.dt)

    def update_voltage(self, x):
        self.v_cell = sf._if_voltage_update(self.v_cell, x, self.alpha_v, self.refrac_counts)

    def forward(self, x):
        self.update_trace(x)
        self.update_voltage(x)
        spikes = self.spiking()
        self.refrac(spikes)
        return spikes


#########################################################
# LIF Neuron
#########################################################
class LIFNeuronTrace(Neuron):
    r"""Leaky Integrate and Fire neuron."""
    def __init__(self,
                 cells_shape,
                 thresh,
                 v_rest,
                 alpha_v,
                 alpha_t,
                 dt,
                 duration_refrac,  # From here on class specific params
                 tau_v,
                 tau_t):
        super(LIFNeuronTrace, self).__init__(cells_shape, thresh, v_rest, alpha_v, alpha_t, dt, duration_refrac)

        # Fixed parameters
        self.tau_v = Parameter(torch.tensor(tau_v, dtype=torch.float))
        self.tau_t = Parameter(torch.tensor(tau_t, dtype=torch.float))
        self.init_neuron()

    def update_trace(self, x):
        self.trace = sf._neuron_exponential_trace_update(self.trace, x, self.alpha_t, self.tau_t, self.dt)

    def update_voltage(self, x):
        self.v_cell = sf._lif_voltage_update(self.v_cell, self.v_rest, x, self.alpha_v, self.tau_v,
            self.dt, self.refrac_counts)

    def forward(self, x):
        self.update_trace(x)
        self.update_voltage(x)
        spikes = self.spiking()
        self.refrac(spikes)
        return spikes


#########################################################
# Fede Neuron
#########################################################
class FedeNeuronTrace(Neuron):
    r"""Leaky Integrate and Fire neuron.

    Defined in "Unsupervised Learning of a Hierarchical Spiking
    Neural Network for Optical Flow Estimation: From Events to
    Global Motion Perception - F.P. Valles, et al."
    """
    def __init__(self,
                 cells_shape,
                 thresh,
                 v_rest,
                 alpha_v,
                 alpha_t,
                 dt,
                 duration_refrac,  # From here on class specific params
                 tau_v,
                 tau_t):
        super(FedeNeuronTrace, self).__init__(cells_shape, thresh, v_rest, alpha_v, alpha_t, dt, duration_refrac)

        # Fixed parameters
        self.tau_v = Parameter(torch.tensor(tau_v, dtype=torch.float))
        self.tau_t = Parameter(torch.tensor(tau_t, dtype=torch.float))
        self.init_neuron()

    def update_trace(self, x):
        self.trace = sf._neuron_exponential_trace_update(self.trace, x, self.alpha_t, self.tau_t, self.dt)

    def update_voltage(self, x, pre_trace):
        self.v_cell = sf._fede_voltage_update(self.v_cell, self.v_rest, x, self.alpha_v, self.tau_v,
            self.dt, self.refrac_counts, pre_trace)

    def forward(self, x, pre_trace):
        self.update_trace(x)
        self.update_voltage(x, pre_trace)
        spikes = self.spiking()
        self.refrac(spikes)
        return spikes