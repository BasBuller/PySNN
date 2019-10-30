import torch
import matplotlib.pyplot as plt

from pysnn.network import SNNNetwork
from pysnn.connection import Linear
from pysnn.neuron import LIFNeuron
from pysnn.learning import MSTDPET


# Parameters
inputs = 1
outputs = 1
c_shape = (inputs, outputs)
dt = 1
alpha_v = 0.3
alpha_t = 1.0
thresh = 0.5
v_rest = 0.0
duration_refrac = 3
voltage_decay = 0.8
trace_decay = 0.8
n_dynamics = (
    thresh,
    v_rest,
    alpha_v,
    alpha_t,
    dt,
    duration_refrac,
    voltage_decay,
    trace_decay,
)
n2_dynamics = (thresh * 2, *n_dynamics[1:])
n_in_dynamics = (dt, alpha_t, trace_decay)
batch_size = 1
delay = 0
c_dynamics = (batch_size, dt, delay)
a_pre = 1.0
a_post = 1.0
lr = 0.1
e_trace_decay = 0.8
l_params = (a_pre, a_post, lr, e_trace_decay)


# Network
class SNN(SNNNetwork):
    def __init__(self):
        super(SNN, self).__init__()

        # One layer
        self.pre_neuron = LIFNeuron((batch_size, 1, inputs), *n_dynamics)
        self.post_neuron = LIFNeuron((batch_size, 1, outputs), *n2_dynamics)
        self.linear = Linear(*c_shape, *c_dynamics)
        self.add_layer("layer1", self.linear, self.post_neuron)

    def forward(self, x):
        pre_spikes, pre_trace = self.pre_neuron(x)
        x, _ = self.linear(pre_spikes, pre_trace)
        post_spikes, post_trace = self.post_neuron(x)

        return pre_spikes, post_spikes, pre_trace, post_trace


if __name__ == "__main__":
    current = []
    pre_spikes = []
    post_spikes = []
    pre_trace = []
    post_trace = []
    rewards = []
    e_trace = []
    weight = []

    # Setup network
    network = SNN()
    network.linear.reset_weights("constant", 3)

    # Determine layers and init learning rule
    layers = network.layer_state_dict()
    learning_rule = MSTDPET(layers, *l_params)

    for i in range(100):
        # Generate input spikes
        curr = (torch.rand(1, 1, 1) > 0.4).float()
        current.append(curr.item())

        # Do forward pass
        pre_s, post_s, pre_t, post_t = network.forward(curr)
        learning_rule.update_state()

        # Append network spikes and traces
        pre_spikes.append(pre_s.item())
        post_spikes.append(post_s.item())
        pre_trace.append(pre_t.item())
        post_trace.append(post_t.item())

        # Do backward/learning pass
        if i < 50:
            reward = 1.0
        else:
            reward = -1.0
        learning_rule.step(reward)

        # Append last resulting items
        rewards.append(reward)
        e_trace.append(learning_rule.layers["layer1"]["e_trace"].item())
        weight.append(network.linear.weight.item())

    # Reset states
    network.reset_state()
    learning_rule.reset_state()

    fig, axs = plt.subplots(8, 1)

    for ax, data in zip(
        axs,
        [
            current,
            pre_spikes,
            post_spikes,
            pre_trace,
            post_trace,
            rewards,
            e_trace,
            weight,
        ],
    ):
        ax.plot(data)

    plt.show()
