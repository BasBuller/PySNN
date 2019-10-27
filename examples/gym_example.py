import gym
import torch
import pytorch_lightning as pl
from tensorboardX import SummaryWriter
from hyperopt import hp
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

from pysnn.network import SNNNetwork
from pysnn.neuron import Input, AdaptiveLIFNeuron, LIFNeuron
from pysnn.connection import Linear
from pysnn.learning import MSTDPET


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
        self.linear1 = Linear(8, 64, *c_dynamics)
        self.linear2 = Linear(64, 1, *c_dynamics)

        # Layers (for learning rule)
        # Consist of connection and post-synaptic neuron
        self.add_layer("fc1", self.linear1, self.neuron1)
        self.add_layer("fc2", self.linear2, self.neuron2)

    def forward(self, x):
        # Encode before passing to input layer
        # Input neuron doesn't return spikes but just the current
        x = self._encode(x)
        x, trace = self.neuron0(x)

        # Hidden layer
        # Connection trace (2nd argument) is not used
        x, _ = self.linear1(x, trace)
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
        return x.abs()

    def _decode(self, x):
        return x.byte()

    def training_step(self, *args, **kwargs):
        # Observation is first positional argument
        # Environment is second
        # Learning rule is third
        # TODO: or step away from args, or use kwargs? (this is just for PL convention)
        obs = torch.from_numpy(args[0]).view(1, 1, -1)
        action = self.forward(obs)

        # Optional render of environment
        if kwargs.get("render", False):
            args[1].render()

        # Do environment step
        action = action.item()
        obs, reward, done, _ = args[1].step(action)

        # Do learning step
        args[2].update_state()
        args[2].step(reward)

        # Return stepped environment and its returns
        # TODO: and learning rule, or are these and env also updated when not returned?
        return obs, reward, done, args[1], args[2], action

    def configure_optimizers(self):
        raise NotImplementedError("PyTorch Lightning not yet fully implemented")


########################################################
# Main
########################################################
def main(config):
    # Put config in neuron/connection/rule dicts
    neuron_in = [config["dt"], config["alpha_t0"], config["tau_t0"]]
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

    # Log with tensorboardX
    # logger = SummaryWriter()
    # logger.add_graph(network, torch.from_numpy(obs).view(1, 1, -1))
    episode_reward = 0.0
    episodes = 0
    actions = 0

    # Simulation loop
    for step in range(config["steps"]):
        obs, reward, done, env, rule, action = network.training_step(
            obs, env, rule, render=False
        )
        actions += action
        episode_reward += reward

        # Episode end
        if done:
            obs = env.reset()
            network.reset_state()
            rule.reset_state()
            # logger.add_scalar("average action", actions / episode_reward, episodes)
            # logger.add_scalar("reward", episode_reward, episodes)
            tune.track.log(reward=episode_reward)
            actions = 0
            episode_reward = 0.0
            episodes += 1

    # Cleanup
    # logger.close()
    env.close()


########################################################
# Entry point
########################################################
if __name__ == "__main__":
    # Input neuron: dt, alpha_t, tau_t
    # n_in_dynamics = [1.0, 1.0, 0.8]
    # Hidden neuron: thresh, v_rest, alpha_v, alpha_t, dt, refrac, tau_v, tau_t, alpha_thresh, tau_thresh
    # n_hid_dynamics = [0.4, 0.0, 0.2, 1.0, 1.0, 0, 0.8, 0.8, 0.2, 0.9]
    # Output neuron: thresh, v_rest, alpha_v, alpha_t, dt, refrac, tau_v, tau_t
    # n_out_dynamics = [0.8, 0.0, 0.2, 1.0, 1.0, 0, 0.8, 0.8]
    # Connections: batch_size, dt, delay
    # c_dynamics = [1, 1.0, 0]
    # Learning rule: a_pre, a_post, lr, tau_e_trace
    # l_params = [1.0, 0.0, 1e-6, 0.9]

    # Parameters in one big dictionary
    # config = {
    #     "network": [n_in_dynamics, n_hid_dynamics, n_out_dynamics, c_dynamics],
    #     "learning rule": l_params,
    #     "steps": 100000,
    # }
    config = {
        "dt": 1.0,
        "thresh1": 0.2,
        "thresh2": hp.uniform("thresh2", 0.0, 1.0),
        "alpha_v1": hp.uniform("alpha_v1", 0.0, 2.0),
        "alpha_v2": hp.uniform("alpha_v2", 0.0, 2.0),
        "alpha_t0": hp.uniform("alpha_t0", 0.0, 2.0),
        "alpha_t1": hp.uniform("alpha_t1", 0.0, 2.0),
        "alpha_t2": hp.uniform("alpha_t2", 0.0, 2.0),
        "alpha_thresh1": hp.uniform("alpha_thresh1", 0.0, 2.0),
        "refrac": 0,
        "v_rest": 0.0,
        "tau_v1": hp.uniform("tau_v1", 0.0, 1.0),
        "tau_v2": hp.uniform("tau_v2", 0.0, 1.0),
        "tau_t0": hp.uniform("tau_t0", 0.0, 1.0),
        "tau_t1": hp.uniform("tau_t1", 0.0, 1.0),
        "tau_t2": hp.uniform("tau_t2", 0.0, 1.0),
        "tau_thresh1": hp.uniform("tau_thresh1", 0.0, 1.0),
        "batch_size": 1,
        "delay": 0,
        "a_pre": hp.uniform("a_pre", 0.0, 2.0),
        "a_post": 0.0,
        "lr": hp.choice("lr", [1e-6, 1e-4, 1e-8, 1e-2]),
        "tau_e_trace": hp.uniform("tau_e_trace", 0.0, 1.0),
        "steps": 10000,
    }

    # Call to main
    # main(config)

    # Run hyperparameter search
    hyperopt_search = HyperOptSearch(config, max_concurrent=8, reward_attr="reward")
    tune.run(
        main,
        num_samples=30,
        scheduler=ASHAScheduler(metric="reward", mode="max"),
        search_alg=hyperopt_search,
        config=config,
    )
