import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from snn.utils import _set_no_grad


#########################################################
# Input neuron
#########################################################
class InputNeuron(nn.Module):
    r"""Input neuron, feeds through all its input to the next layer.
    
    The reason this class exists is to allow for learning weights 
    from the input to the first real layer of the network. 
    """
    def __init__(self, n_cells):
        super(InputNeuron, self).__init__() 

    def forward(self, x):
        return x > 0


#########################################################
# Neuron
#########################################################
class Neuron(nn.Module):
    r"""Base neuron model, is a container to define basic neuron functionalties. 

    Defines basic spiking, voltage and trace characteristics. Just has to adhere 
    to the API functionalities to integrate within Connection modules.

    Make sure the Neuron class receives input voltage for each neuron and returns a 
    Tensor indicating which neurons have spiked.
    """
    def __init__(self, 
                 n_cells, 
                 thresh, 
                 v_rest, 
                 alpha, 
                 dt, 
                 duration_refrac):
        super(Neuron, self).__init__()
        self.n_cells = n_cells

        # Fixed parameters
        self.v_rest = v_rest
        self.alpha = alpha
        self.dt = dt
        self.duration_refrac = duration_refrac
        self.thresh_center = thresh

        # Define dynamic parameters
        self.v_cell = Parameter(torch.Tensor(1, n_cells))
        self.refrac_counts = Parameter(torch.Tensor(1, n_cells))

        # Define learnable parameters
        self.thresh = Parameter(torch.Tensor(1, n_cells))

        # Initialize parameters
        self.reset_state()
        self.reset_parameters()
        self.no_grad()

    def spiking(self):
        return self.v_cell > self.thresh

    def refrac(self, spikes):
        self.refrac_counts[self.refrac_counts > 0] -= self.dt
        self.v_cell[spikes] = 0
        self.refrac_counts[spikes] = self.duration_refrac
        self.refrac_counts.clamp_(0, self.duration_refrac)

    def reset_state(self):
        self.v_cell.data.fill_(self.v_rest)
        self.refrac_counts.data.fill_(0)

    def reset_parameters(self):
        self.thresh.data = torch.ones_like(self.thresh.data) * self.thresh_center

    def no_grad(self):
        _set_no_grad(self)
        self.train(False)


#########################################################
# IF Neuron
#########################################################
class IFNeuron(Neuron):
    r"""Integrate and Fire neuron."""
    def __init__(self, 
                 n_cells, 
                 thresh, 
                 v_rest, 
                 alpha, 
                 dt, 
                 duration_refrac):
        super(IFNeuron, self).__init__(n_cells, thresh, v_rest, alpha, dt, duration_refrac)

    def update_voltage(self, x):
        v_delta = self.alpha * x
        non_refrac = self.refrac_counts == 0
        self.v_cell[non_refrac] += v_delta[non_refrac]

    def forward(self, x):
        self.update_voltage(x)
        spikes = self.spiking()
        self.refrac(spikes)
        return spikes


#########################################################
# LIF Neuron
#########################################################
class LIFNeuron(Neuron):
    r"""Leaky Integrate and Fire neuron."""
    def __init__(self, 
                 n_cells, 
                 thresh, 
                 v_rest, 
                 alpha, 
                 dt, 
                 duration_refrac,  # From here on class specific params
                 tau_v):
        super(LIFNeuron, self).__init__(n_cells, thresh, v_rest, alpha, dt, duration_refrac)
        self.tau_v = tau_v

    def update_voltage(self, x):
        v_delta = (-(self.v_cell - self.v_rest) * self.dt + self.alpha * x) / self.tau_v
        non_refrac = self.refrac_counts == 0
        self.v_cell[non_refrac] += v_delta[non_refrac]

    def forward(self, x):
        self.update_voltage(x)
        spikes = self.spiking()
        self.refrac(spikes)
        return spikes
