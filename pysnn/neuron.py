import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from pysnn.utils import _set_no_grad
import pysnn.functional as sf


######################################################### 
# Input Neuron
#########################################################
class Input(nn.Module):
    r"""Simple feed-through layer of neurons used for storing a trace."""
    def __init__(self, cells_shape, dt):
        super(Input, self).__init__()
        self.trace = Parameter(torch.zeros(*cells_shape, dtype=torch.float))
        self.dt = Parameter(torch.tensor(dt, dtype=torch.float))

    def reset_state(self):
        r"""Reset cell states that accumulate over time during simulation."""
        self.trace.fill_(0)

    def no_grad(self):
        r"""Turn off learning and gradient storing."""
        _set_no_grad(self)

    def init_neuron(self):
        r"""Initialize state, parameters and turn off gradients."""
        self.no_grad()
        self.reset_state()

    def convert_input(self, x):
        return x.type(self.trace.dtype)


class InputTraceExponential(Input):
    def __init__(self, cells_shape, dt, alpha_t, tau_t):
        super(InputTraceExponential, self).__init__(cells_shape, dt)
        self.alpha_t = torch.tensor(alpha_t, dtype=torch.float)
        self.tau_t = torch.tensor(tau_t, dtype=torch.float)

        self.init_neuron()

    def update_trace(self, x):
        x = self.convert_input(x)
        self.trace = sf._exponential_trace_update(self.trace, x, self.alpha_t, self.tau_t, self.dt)

    def forward(self, x):
        self.update_trace(x)
        return x, self.trace


class InputTraceLinear(Input):
    def __init__(self, cells_shape, dt, alpha_t, trace_decay):
        super(InputTraceLinear, self).__init__(cells_shape, dt)
        self.alpha_t = torch.tensor(alpha_t, dtype=torch.float)
        self.trace_decay = torch.tensor(trace_decay, dtype=torch.float)

        self.init_neuron()

    def update_trace(self, x):
        x = self.convert_input(x)
        self.trace = sf._linear_trace_update(self.trace, x, self.alpha_t, self.trace_decay)

    def forward(self, x):
        self.update_trace(x)
        return x, self.trace


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
                 duration_refrac,
                 store_trace=False):
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

        # In case of storing a complete, local copy of the activity of a neuron
        if store_trace:
            # self.complete_trace = Parameter(torch.zeros(*cells_shape, 1, dtype=torch.bool, requires_grad=False).to(torch.bool))
            self.complete_trace = Parameter(torch.zeros(*cells_shape, 1))
            self.complete_trace.requires_grad = False
            self.complete_trace.data = self.complete_trace.to(torch.bool)
        else:
            self.complete_trace = None

    def spiking(self):
        r"""Return cells that are in spiking state."""
        return self.v_cell >= self.thresh

    def refrac(self, spikes):
        r"""Basic counting version of cell refractory period.

        Can be overwritten in case of the need of more refined functionality.
        """
        self.refrac_counts[self.refrac_counts > 0] -= self.dt
        self.refrac_counts += self.duration_refrac * self.convert_spikes(spikes)
        self.v_cell.masked_fill_(spikes, self.v_rest)

    def concat_trace(self, x):
        r"""Concatenate most recent timestep to the trace storage."""
        self.complete_trace.data = torch.cat([self.complete_trace, x.unsqueeze(-1)], dim=-1)

    def fold(self, x):
        r"""Fold incoming spike train by summing last dimension."""
        return x.sum(-1)

    def unfold(self, x):
        r"""Move the last dimension (all incoming to single neuron in current layer) to first dim.

        This is done because PyTorch broadcasting does not support broadcasting over the last dim.
        """
        shape = x.shape
        return x.view(shape[-1], *shape[:-1])

    def convert_spikes(self, spikes):
        r"""Cast uint8 spikes to datatype that is used for voltage and weights"""
        return spikes.to(self.v_cell.dtype)

    def reset_state(self):
        r"""Reset cell states that accumulate over time during simulation."""
        self.v_cell.fill_(self.v_rest)
        self.refrac_counts.fill_(0)
        self.trace.fill_(0)
        if self.complete_trace is not None:
            self.complete_trace.data = torch.zeros(*self.v_cell.shape, 1, device=self.v_cell.device).to(torch.bool)

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
class IFNeuronTraceLinear(Neuron):
    r"""Integrate and Fire neuron."""
    def __init__(self,
                 cells_shape,
                 thresh,
                 v_rest,
                 alpha_v,
                 alpha_t,
                 dt,
                 duration_refrac,
                 tau_t,
                 store_trace=False):
        super(IFNeuronTraceLinear, self).__init__(cells_shape, thresh, v_rest, alpha_v, alpha_t, 
                                                  dt, duration_refrac, store_trace=store_trace)

        #Fixed parameters
        self.tau_t = Parameter(torch.tensor(tau_t, dtype=torch.float))
        self.init_neuron()

    def update_trace(self, x):
        spikes = self.convert_spikes(x)
        self.trace = sf._linear_trace_update(self.trace, spikes, self.alpha_t, self.tau_t)

    def update_voltage(self, x):
        self.v_cell = sf._if_voltage_update(self.v_cell, x, self.alpha_v, self.refrac_counts)

    def forward(self, x):
        x = self.fold(x)
        self.update_voltage(x)
        spikes = self.spiking()
        self.update_trace(spikes)
        self.refrac(spikes)
        return spikes, self.trace


class IFNeuronTraceExponential(Neuron):
    r"""Integrate and Fire neuron."""
    def __init__(self,
                 cells_shape,
                 thresh,
                 v_rest,
                 alpha_v,
                 alpha_t,
                 dt,
                 duration_refrac,
                 tau_t,
                 store_trace=False):
        super(IFNeuronTraceExponential, self).__init__(cells_shape, thresh, v_rest, alpha_v, alpha_t, 
                                                       dt, duration_refrac, store_trace=store_trace)

        #Fixed parameters
        self.tau_t = Parameter(torch.tensor(tau_t, dtype=torch.float))
        self.init_neuron()

    def update_trace(self, x):
        spikes = self.convert_spikes(x)
        self.trace = sf._exponential_trace_update(self.trace, spikes, self.alpha_t, self.tau_t, self.dt)

    def update_voltage(self, x):
        self.v_cell = sf._if_voltage_update(self.v_cell, x, self.alpha_v, self.refrac_counts)

    def forward(self, x):
        x = self.fold(x)
        self.update_voltage(x)
        spikes = self.spiking()
        self.update_trace(spikes)
        self.refrac(spikes)
        return spikes, self.trace


#########################################################
# LIF Neuron
#########################################################
class LIFNeuronTraceLinear(Neuron):
    r"""Leaky Integrate and Fire neuron."""
    def __init__(self,
                 cells_shape,
                 thresh,
                 v_rest,
                 alpha_v,
                 alpha_t,
                 dt,
                 duration_refrac,  # From here on class specific params
                 voltage_decay,
                 trace_decay,
                 store_trace=False):
        super(LIFNeuronTraceLinear, self).__init__(cells_shape, thresh, v_rest, alpha_v, alpha_t, 
                                                   dt, duration_refrac, store_trace=store_trace)

        # Fixed parameters
        self.voltage_decay = torch.tensor(voltage_decay, dtype=torch.float)
        self.trace_decay = torch.tensor(trace_decay, dtype=torch.float)
        self.init_neuron()

    def update_trace(self, x):
        spikes = self.convert_spikes(x)
        self.trace = sf._linear_trace_update(self.trace, spikes, self.alpha_t, self.trace_decay)

    def update_voltage(self, x):
        self.v_cell.data = sf._lif_linear_voltage_update(self.v_cell, self.v_rest, x, self.alpha_v, self.voltage_decay,
            self.dt, self.refrac_counts)

    def forward(self, x):
        x = self.fold(x)
        self.update_voltage(x)
        spikes = self.spiking()
        self.update_trace(spikes)
        self.refrac(spikes)
        if self.complete_trace is not None:
            self.concat_trace(spikes)
        return spikes, self.trace


class LIFNeuronTraceExponential(Neuron):
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
                 tau_t,
                 store_trace=False):
        super(LIFNeuronTraceExponential, self).__init__(cells_shape, thresh, v_rest, alpha_v, alpha_t, 
                                                        dt, duration_refrac, store_trace=store_trace)

        # Fixed parameters
        self.tau_v = Parameter(torch.tensor(tau_v, dtype=torch.float))
        self.tau_t = Parameter(torch.tensor(tau_t, dtype=torch.float))
        self.init_neuron()

    def update_trace(self, x):
        spikes = self.convert_spikes(x)
        self.trace = sf._exponential_trace_update(self.trace, spikes, self.alpha_t, self.tau_t, self.dt)

    def update_voltage(self, x):
        self.v_cell = sf._lif_voltage_update(self.v_cell, self.v_rest, x, self.alpha_v, self.tau_v,
            self.dt, self.refrac_counts)

    def forward(self, x):
        x = self.fold(x)
        self.update_voltage(x)
        spikes = self.spiking()
        self.update_trace(spikes)
        self.refrac(spikes)
        return spikes, self.trace


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
                 tau_t,
                 store_trace=False):
        super(FedeNeuronTrace, self).__init__(cells_shape, thresh, v_rest, alpha_v, alpha_t, 
                                              dt, duration_refrac, store_trace=store_trace)

        # Fixed parameters
        self.tau_v = Parameter(torch.tensor(tau_v, dtype=torch.float))
        self.tau_t = Parameter(torch.tensor(tau_t, dtype=torch.float))
        self.init_neuron()

    def update_trace(self, x):
        x = self.convert_spikes(x)
        self.trace = sf._exponential_trace_update(self.trace, x, self.alpha_t, self.tau_t, self.dt)

    def update_voltage(self, x, pre_trace):
        self.v_cell = sf._fede_voltage_update(self.v_cell, self.v_rest, x, self.alpha_v, self.tau_v,
            self.dt, self.refrac_counts, pre_trace)

    def forward(self, x, pre_trace):
        # x = self.fold(x)
        self.update_voltage(x, pre_trace)
        spikes = self.spiking()
        self.update_trace(spikes)
        self.refrac(spikes)
        if self.complete_trace is not None:
            self.concat_trace(spikes)
        return spikes, self.trace
