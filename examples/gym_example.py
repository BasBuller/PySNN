import gym
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch

from pysnn.network import SNNNetwork
from pysnn.neuron import AdaptiveLIFNeuron, LIFNeuron
from pysnn.connection import Linear
from pysnn.learning import MSTDPET


########################################################
# Spiking network
########################################################
class SpikingNet(SNNNetwork):
    def __init__(self, n_in_dynamics, n_hid_dynamics, n_out_dynamics, c_dynamics):
        super(SpikingNet, self).__init__()

        # Input layer
        # 4-dimensional observation of + and -, so 8 neurons
        self.neuron0 = AdaptiveLIFNeuron((1, 1, 8), *n_in_dynamics)

        # Hidden layer
        # Adaptive neuron to cope with highly varying input
        self.neuron1 = AdaptiveLIFNeuron((1, 1, 64), *n_hid_dynamics)

        # Output layer
        # Non-adaptive neuron to decrease latency in action selection
        # Action is binary, so single neuron suffices
        self.neuron2 = LIFNeuron((1, 1, 1), *n_out_dynamics)

        # Connections
        # Weights initialized uniformly in [0, 1]
        self.linear1 = Linear(8, 64, *c_dynamics)
        self.linear2 = Linear(64, 1, *c_dynamics)

        # Layers (for learning rule)
        # Consist of connection and post-synaptic neuron
        self.add_layer("fc1", self.linear1, self.neuron1)
        self.add_layer("fc2", self.linear2, self.neuron2)

    def forward(self, x):
        # Encode before passing to input layer
        x = self._encode(x)
        spikes, trace = self.neuron0(x)

        # Hidden layer
        # Connection trace (2nd argument) is not used
        x, _ = self.linear1(spikes, trace)
        spikes, trace = self.neuron1(x)

        # Output layer
        x, _ = self.linear2(spikes, trace)
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
        return x.abs().float()

    def _decode(self, x):
        return x.byte()

    def step(self, obs, env, rule, render=False):
        # Observation is first positional argument
        # Environment is second
        # Learning rule is third
        # TODO: or step away from args, or use kwargs? (this is just for PL convention)
        obs = torch.from_numpy(obs).view(1, 1, -1)
        action = self.forward(obs)

        # Optional render of environment
        if render:
            env.render()

        # Do environment step
        action = action.item()
        obs, reward, done, _ = env.step(action)

        # Do learning step
        rule.update_state()
        rule.step(reward)

        # Return stepped environment and its returns
        return obs, reward, done, env, rule


########################################################
# Main
########################################################
def main(config):
    # Put config in neuron/connection/rule dicts
    neuron_in = [
        config["thresh0"],
        config["v_rest"],
        config["alpha_v0"],
        config["alpha_t0"],
        config["dt"],
        config["refrac"],
        config["tau_v0"],
        config["tau_t0"],
        config["alpha_thresh0"],
        config["tau_thresh0"],
    ]
    neuron_hid = [
        config["thresh1"],
        config["v_rest"],
        config["alpha_v1"],
        config["alpha_t1"],
        config["dt"],
        config["refrac"],
        config["tau_v1"],
        config["tau_t1"],
        config["alpha_thresh1"],
        config["tau_thresh1"],
    ]
    neuron_out = [
        config["thresh2"],
        config["v_rest"],
        config["alpha_v2"],
        config["alpha_t2"],
        config["dt"],
        config["refrac"],
        config["tau_v2"],
        config["tau_t2"],
    ]
    conns = [config["batch_size"], config["dt"], config["delay"]]
    lr = [config["a_pre"], config["a_post"], config["lr"], config["tau_e_trace"]]

    # Build network
    # Build learning rule from network layers
    network = SpikingNet(neuron_in, neuron_hid, neuron_out, conns)
    rule = MSTDPET(network.layer_state_dict(), *lr)

    # Build env
    env = gym.make("CartPole-v1")
    obs = env.reset()

    # Logging variables
    episode_reward = 0.0

    # Simulation loop
    for step in range(config["steps"]):
        obs, reward, done, env, rule = network.step(obs, env, rule, render=False)
        episode_reward += reward

        # Episode end
        if done:
            obs = env.reset()
            network.reset_state()
            rule.reset_state()
            tune.track.log(reward=episode_reward)
            episode_reward = 0.0

    # Cleanup
    env.close()


if __name__ == "__main__":
    # Fixed parameters
    config = {
        "dt": 1.0,
        "thresh0": 0.2,
        "thresh1": 0.2,
        "refrac": 0,
        "v_rest": 0.0,
        "batch_size": 1,
        "delay": 0,
        "a_post": 0.0,
        "steps": 10000,
    }

    # Search space for Bayesian Optimization
    space = {
        "thresh2": (0.0, 1.0),
        "alpha_v0": (0.0, 2.0),
        "alpha_v1": (0.0, 2.0),
        "alpha_v2": (0.0, 2.0),
        "alpha_t0": (0.0, 2.0),
        "alpha_t1": (0.0, 2.0),
        "alpha_t2": (0.0, 2.0),
        "alpha_thresh0": (0.0, 2.0),
        "alpha_thresh1": (0.0, 2.0),
        "tau_v0": (0.0, 1.0),
        "tau_v1": (0.0, 1.0),
        "tau_v2": (0.0, 1.0),
        "tau_t0": (0.0, 1.0),
        "tau_t1": (0.0, 1.0),
        "tau_t2": (0.0, 1.0),
        "tau_thresh0": (0.0, 1.0),
        "tau_thresh1": (0.0, 1.0),
        "a_pre": (0.0, 2.0),
        "lr": (1e-6, 1e-2),
        "tau_e_trace": (0.0, 1.0),
    }

    # Run hyperparameter search
    search = BayesOptSearch(
        space,
        max_concurrent=6,
        metric="reward",
        mode="max",
        utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0},
    )
    scheduler = ASHAScheduler(metric="reward", mode="max")
    tune.run(
        main,
        num_samples=100,
        scheduler=scheduler,
        search_alg=search,
        config=config,
        verbose=1,
        local_dir="ray_runs",
    )
