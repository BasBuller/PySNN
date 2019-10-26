import torch
import pytorch_lightning as pl

from pysnn.network import SNNNetwork
from pysnn.neuron import Input, AdaptiveLIFNeuron, LIFNeuron
from pysnn.connection import Linear

########################################################
# Spiking network
########################################################


class SpikingNet(SNNNetwork, pl.LightningModule):
    def __init__(self, n_in_dynamics, n_hid_dynamics, n_out_dynamics, c_dynamics):
        super(SpikingNet, self).__init__()

        # Input layer
        # 4-dimensional observation of + and -, so 8 neurons
        self.neuron0 = Input((1, 1, 8), *n_in_dynamics)

        # Hidden layer
        # Adaptive neuron to cope with highly varying input
        self.neuron1 = AdaptiveLIFNeuron((1, 1, 64), *n_hid_dynamics)

        # Output layer
        # Non-adaptive neuron to decrease latency in action selection
        # Action is binary, so single neuron suffices
        self.neuron2 = LIFNeuron((1, 1, 1), *n_out_dynamics)

        # Connections
        # Weights initialized uniformly in [0, 1]
        self.fc1 = Linear(8, 64, *c_dynamics)
        self.fc2 = Linear(64, 1, *c_dynamics)

    def forward(self, x):
        # Encode before passing to input layer
        # Input neuron doesn't return spikes but just the current
        x = self._encode(x)
        x, trace = self.neuron0(x)

        # Hidden layer
        # Connection trace (2nd argument) is not used
        x, _ = self.fc1(x, trace)
        spikes, trace = self.neuron1(x)

        # Output layer
        x, _ = self.fc2(spikes, trace)
        spikes, trace = self.neuron2(x)

        # Decode into binary action (so nothing)
        return self._decode(spikes)

    def _encode(self, x):
        # Repeat the input
        x = x.repeat(1, 1, 2)

        # Clamp first half to positive, second to negative
        x[..., :4].clamp_(min=0)
        x[..., 4:].clamp_(max=0)

        # Make absolute
        return x.abs()

    def _decode(self, x):
        return x.view(-1)

    def training_step(self, *args, **kwargs):
        raise NotImplementedError("PyTorch Lightning not yet working")

    def configure_optimizers(self):
        raise NotImplementedError("How to do this for spiking nets?")


########################################################
# Simulation loop
########################################################


########################################################
# Main
########################################################


def main():
    pass


########################################################
# Entry point
########################################################

if __name__ == "__main__":
    main()
