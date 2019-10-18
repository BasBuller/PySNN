import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from pysnn.utils import _set_no_grad
import pysnn.functional as sf


#########################################################
# Input Neuron
#########################################################
class BaseInput(nn.Module):
    r"""Simple feed-through layer of neurons used for storing a trace."""

    def __init__(self, cells_shape, dt):
        super(BaseInput, self).__init__()
        self.register_buffer("trace", torch.zeros(*cells_shape, dtype=torch.float))
        self.register_buffer("dt", torch.tensor(dt, dtype=torch.float))

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

    def forward(self, x):
        raise NotImplementedError("Input neurons must implement `forward`")

    def update_trace(self, x):
        raise NotImplementedError("Input neurons must implement `update_trace`")


class Input(BaseInput):
    def __init__(self, cells_shape, dt, alpha_t, tau_t, update_type="linear"):
        super(Input, self).__init__(cells_shape, dt)
        self.register_buffer("alpha_t", torch.tensor(alpha_t, dtype=torch.float))
        self.register_buffer("tau_t", torch.tensor(tau_t, dtype=torch.float))

        # Type of updates
        if update_type == "linear":
            self.trace_update = sf._linear_trace_update
        elif update_type == "exponential":
            self.trace_update = sf._exponential_trace_update
        else:
            raise ValueError(f"Unsupported trace type {update_type}")

        self.init_neuron()

    def update_trace(self, x):
        x = self.convert_input(x)
        self.trace = self.trace_update(self.trace, x, self.alpha_t, self.tau_t, self.dt)

    def forward(self, x):
        self.update_trace(x)
        return x, self.trace


#########################################################
# Base Neuron
#########################################################
class BaseNeuron(nn.Module):
    r"""Base neuron model, is a container to define basic neuron functionalties.

    Defines basic spiking, voltage and trace characteristics. Just has to
    adhere to the API functionalities to integrate within Connection modules.

    Make sure the Neuron class receives input voltage for each neuron and
    returns a Tensor indicating which neurons have spiked.
    """

    def __init__(
        self,
        cells_shape,
        thresh,
        v_rest,
        alpha_v,
        alpha_t,
        dt,
        duration_refrac,
        store_trace=False,
    ):
        super(BaseNeuron, self).__init__()
        self.cells_shape = torch.tensor(cells_shape)

        # Check compatibility of dt and refrac counting
        assert (
            duration_refrac % dt == 0
        ), "dt does not fit an integer amount of times in duration_refrac"

        # Fixed parameters
        self.register_buffer("v_rest", torch.tensor(v_rest, dtype=torch.float))
        self.register_buffer(
            "alpha_v", torch.tensor(alpha_v, dtype=torch.float)
        )  # TODO: Might want to move this out of base class
        self.register_buffer(
            "alpha_t", torch.tensor(alpha_t, dtype=torch.float)
        )  # TODO: Might want to move this out of base class
        self.register_buffer("dt", torch.tensor(dt, dtype=torch.float))
        self.register_buffer(
            "duration_refrac", torch.tensor(duration_refrac + 1, dtype=torch.float)
        )
        self.register_buffer("thresh_center", torch.tensor(thresh, dtype=torch.float))

        # Define dynamic parameters
        self.register_buffer("spikes", torch.Tensor(*cells_shape))
        self.register_buffer("v_cell", torch.Tensor(*cells_shape))
        self.register_buffer("trace", torch.Tensor(*cells_shape))
        self.register_buffer("refrac_counts", torch.Tensor(*cells_shape))

        # Define learnable parameters
        self.thresh = Parameter(torch.Tensor(*cells_shape))

        # In case of storing a complete, local copy of the activity of a neuron
        if store_trace:
            complete_trace = torch.zeros(*cells_shape, 1, dtype=torch.bool)
        else:
            complete_trace = None
        self.register_buffer("complete_trace", complete_trace)

    def spiking(self):
        r"""Return cells that are in spiking state."""
        self.spikes = self.v_cell >= self.thresh
        return self.spikes

    def refrac(self, spikes):
        r"""Basic counting version of cell refractory period.

        Can be overwritten in case of the need of more refined functionality.
        """
        if self.duration_refrac > 1:
            self.refrac_counts[self.refrac_counts > 0] -= self.dt
            self.refrac_counts += self.duration_refrac * self.convert_spikes(spikes)
        else:
            self.refrac_counts.copy_(self.convert_spikes(spikes))
        self.v_cell.masked_fill_(spikes, self.v_rest)

    def concat_trace(self, x):
        r"""Concatenate most recent timestep to the trace storage."""
        self.complete_trace = torch.cat([self.complete_trace, x.unsqueeze(-1)], dim=-1)

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
        self.spikes.fill_(False)
        self.refrac_counts.fill_(0)
        self.trace.fill_(0)
        if self.complete_trace is not None:
            self.complete_trace = torch.zeros(
                *self.v_cell.shape, 1, device=self.v_cell.device
            ).to(torch.bool)

    def reset_thresh(self):
        r"""Reset threshold to initialization values, allows for different standard thresholds per neuron."""
        self.thresh.copy_(torch.ones_like(self.thresh) * self.thresh_center)

    def no_grad(self):
        r"""Turn off learning and gradient storing."""
        _set_no_grad(self)

    def init_neuron(self):
        r"""Initialize state, parameters and turn off gradients."""
        self.no_grad()
        self.reset_state()
        self.reset_thresh()

    def forward(self, x):
        raise NotImplementedError("Neurons must implement `forward`")

    def update_trace(self, x):
        raise NotImplementedError("Neurons must implement `update_trace`")

    def update_voltage(self, x):
        raise NotImplementedError("Neurons must implement `update_voltage`")


#########################################################
# IF Neuron
#########################################################
class IFNeuron(BaseNeuron):
    r"""Integrate and Fire neuron."""

    def __init__(
        self,
        cells_shape,
        thresh,
        v_rest,
        alpha_v,
        alpha_t,
        dt,
        duration_refrac,
        tau_t,
        update_type="linear",
        store_trace=False,
    ):
        super(IFNeuron, self).__init__(
            cells_shape,
            thresh,
            v_rest,
            alpha_v,
            alpha_t,
            dt,
            duration_refrac,
            store_trace=store_trace,
        )

        # Type of updates
        if update_type == "linear":
            self.trace_update = sf._linear_trace_update
        elif update_type == "exponential":
            self.trace_update = sf._exponential_trace_update
        else:
            raise ValueError(f"Unsupported trace type {update_type}")

        # Fixed parameters
        self.register_buffer("tau_t", torch.tensor(tau_t, dtype=torch.float))
        self.init_neuron()

    def update_trace(self, x):
        spikes = self.convert_spikes(x)
        self.trace = self.trace_update(
            self.trace, spikes, self.alpha_t, self.tau_t, self.dt
        )

    def update_voltage(self, x):
        self.v_cell = sf._if_voltage_update(
            self.v_cell, x, self.alpha_v, self.refrac_counts
        )

    def forward(self, x):
        x = self.fold(x)
        self.update_voltage(x)
        spikes = self.spiking()
        self.update_trace(spikes)
        self.refrac(spikes)
        if self.complete_trace is not None:
            self.concat_trace(spikes)
        return spikes, self.trace


#########################################################
# LIF Neuron
#########################################################
class LIFNeuron(BaseNeuron):
    r"""Leaky Integrate and Fire neuron."""

    def __init__(
        self,
        cells_shape,
        thresh,
        v_rest,
        alpha_v,
        alpha_t,
        dt,
        duration_refrac,  # From here on class specific params
        tau_v,
        tau_t,
        update_type="linear",
        store_trace=False,
    ):
        super(LIFNeuron, self).__init__(
            cells_shape,
            thresh,
            v_rest,
            alpha_v,
            alpha_t,
            dt,
            duration_refrac,
            store_trace=store_trace,
        )

        # Type of updates
        if update_type == "linear":
            self.voltage_update = sf._lif_linear_voltage_update
            self.trace_update = sf._linear_trace_update
        elif update_type == "exponential":
            self.voltage_update = sf._lif_exponential_voltage_update
            self.trace_update = sf._exponential_trace_update
        else:
            raise ValueError(f"Unsupported update type {update_type}")

        # Fixed parameters
        self.register_buffer("tau_v", torch.tensor(tau_v, dtype=torch.float))
        self.register_buffer("tau_t", torch.tensor(tau_t, dtype=torch.float))
        self.init_neuron()

    def update_trace(self, x):
        spikes = self.convert_spikes(x)
        self.trace = self.trace_update(
            self.trace, spikes, self.alpha_t, self.tau_t, self.dt
        )

    def update_voltage(self, x):
        self.v_cell = self.voltage_update(
            self.v_cell,
            self.v_rest,
            x,
            self.alpha_v,
            self.tau_v,
            self.dt,
            self.refrac_counts,
        )

    def forward(self, x):
        x = self.fold(x)
        self.update_voltage(x)
        spikes = self.spiking()
        self.update_trace(spikes)
        self.refrac(spikes)
        if self.complete_trace is not None:
            self.concat_trace(spikes)
        return spikes, self.trace


#########################################################
# Adaptive LIF Neuron
#########################################################
class AdaptiveLIFNeuron(BaseNeuron):
    r"""Adaptive Leaky Integrate and Fire neuron."""

    def __init__(
        self,
        cells_shape,
        thresh,
        v_rest,
        alpha_v,
        alpha_t,
        dt,
        duration_refrac,  # From here on class specific params
        tau_v,
        tau_t,
        alpha_thresh,
        tau_thresh,
        update_type="linear",
        store_trace=False,
    ):
        super(AdaptiveLIFNeuron, self).__init__(
            cells_shape,
            thresh,
            v_rest,
            alpha_v,
            alpha_t,
            dt,
            duration_refrac,
            store_trace=store_trace,
        )

        # Type of updates
        if update_type == "linear":
            self.voltage_update = sf._lif_linear_voltage_update
            self.trace_update = sf._linear_trace_update
            self.thresh_update = sf._linear_thresh_update
        elif update_type == "exponential":
            self.voltage_update = sf._lif_voltage_update
            self.trace_update = sf._exponential_trace_update
            self.thresh_update = sf._exponential_thresh_update
        else:
            raise ValueError(f"Unsupported update type {update_type}")

        # Fixed parameters
        self.register_buffer("tau_v", torch.tensor(tau_v, dtype=torch.float))
        self.register_buffer("tau_t", torch.tensor(tau_t, dtype=torch.float))
        self.register_buffer(
            "alpha_thresh", torch.tensor(alpha_thresh, dtype=torch.float)
        )
        self.register_buffer("tau_thresh", torch.tensor(tau_thresh, dtype=torch.float))
        self.init_neuron()

    def update_trace(self, x):
        spikes = self.convert_spikes(x)
        self.trace = self.trace_update(
            self.trace, spikes, self.alpha_t, self.tau_t, self.dt
        )

    def update_thresh(self, x):
        r"""Return cells that are in spiking state and adjust threshold accordingly."""
        spikes = self.convert_spikes(x)
        self.thresh = self.thresh_update(
            self.thresh, spikes, self.alpha_thresh, self.tau_thresh, self.dt
        )
        # No clamping needed since multiplication!

    def update_voltage(self, x):
        self.v_cell = self.voltage_update(
            self.v_cell,
            self.v_rest,
            x,
            self.alpha_v,
            self.tau_v,
            self.dt,
            self.refrac_counts,
        )

    def forward(self, x):
        x = self.fold(x)
        self.update_voltage(x)
        spikes = self.spiking()
        self.update_trace(spikes)
        self.update_thresh(spikes)
        self.refrac(spikes)
        if self.complete_trace is not None:
            self.concat_trace(spikes)
        return spikes, self.trace


#########################################################
# Fede Neuron
#########################################################
class FedeNeuron(BaseNeuron):
    r"""Leaky Integrate and Fire neuron.

    Defined in "Unsupervised Learning of a Hierarchical Spiking
    Neural Network for Optical Flow Estimation: From Events to
    Global Motion Perception - F.P. Valles, et al."
    """

    def __init__(
        self,
        cells_shape,
        thresh,
        v_rest,
        alpha_v,
        alpha_t,
        dt,
        duration_refrac,  # From here on class specific params
        tau_v,
        tau_t,
        store_trace=False,
    ):
        super(FedeNeuron, self).__init__(
            cells_shape,
            thresh,
            v_rest,
            alpha_v,
            alpha_t,
            dt,
            duration_refrac,
            store_trace=store_trace,
        )

        # Fixed parameters
        self.register_buffer("tau_v", torch.tensor(tau_v, dtype=torch.float))
        self.register_buffer("tau_t", torch.tensor(tau_t, dtype=torch.float))
        self.init_neuron()

    def update_trace(self, x):
        self.trace = sf._exponential_trace_update(
            self.trace, x, self.alpha_t, self.tau_t, self.dt
        )

    def update_voltage(self, x, pre_trace):
        self.v_cell = sf._fede_voltage_update(
            self.v_cell,
            self.v_rest,
            x,
            self.alpha_v,
            self.tau_v,
            self.dt,
            self.refrac_counts,
            pre_trace,
        )

    def forward(self, x, pre_trace):
        self.update_voltage(x, pre_trace)
        spikes = self.spiking()
        self.update_trace(self.convert_spikes(spikes))
        self.refrac(spikes)
        if self.complete_trace is not None:
            self.concat_trace(spikes)
        return spikes, self.trace
