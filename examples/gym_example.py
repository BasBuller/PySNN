import gym
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
        # Observation is first positional argument
        # Environment is second
        # Learning rule is third
        # TODO: or step away from args, or use kwargs? (this is just for PL convention)
        obs = torch.from_numpy(args[0])
        action = self.forward(obs.view(1, 1, -1))
        action = action.numpy()
        obs, reward, done, _ = args[1].step(action)

        # Do learning step
        args[2].update_state()
        args[2].step(reward)

        # Return stepped environment and its returns
        # TODO: and learning rule, or are these and env also updated when not returned?
        return obs, reward, done, args[1], args[2]

    def configure_optimizers(self):
        raise NotImplementedError("PyTorch Lightning not yet fully implemented")


########################################################
# Main
########################################################
def main(config):
    # Build network
    # Add layers for learning rules
    network = SpikingNet(*config["network"])

    # Build env
    env = gym.make("CartPole-v1")
    obs = env.reset()
    done = False

    # Start tensorboard?

    # Tune hypers?

    # Simulation loop
    while not done:
        obs, reward, done, env, rule = network.training_step(obs, env, rule)


########################################################
# Entry point
########################################################
if __name__ == "__main__":
    # Input neuron: dt, alpha_t, tau_t
    n_in_dynamics = [1.0, 1.0, 0.8]
    # Hidden neuron: thresh, v_rest, alpha_v, alpha_t, dt, refrac, tau_v, tau_t, alpha_thresh, tau_thresh
    n_hid_dynamics = [0.4, 0.0, 0.2, 1.0, 1.0, 0, 0.8, 0.8, 0.2, 0.9]
    # Output neuron: thresh, v_rest, alpha_v, alpha_t, dt, refrac, tau_v, tau_t
    n_out_dynamics = [0.8, 0.0, 0.2, 1.0, 1.0, 0, 0.8, 0.8]
    # Connections: batch_size, dt, delay
    c_dynamcis = [1, 1.0, 0]
    # Learning rule: a_pre, a_post, lr, tau_e_trace
    l_params = [1.0, 0.0, 0.01, 0.9]

    # Parameters in one big dictionary
    config = {
        "network": [n_in_dynamics, n_hid_dynamics, n_out_dynamics, c_dynamcis],
        "learning rule": l_params,
        "episodes": 100,
    }

    # Call to main
    main(config)
