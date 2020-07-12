import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from seaborn import heatmap

from pysnn.connection import Linear
from pysnn.neuron import GradLIFNeuron, Input, GradReadoutNeuron
from pysnn.encoding import PoissonEncoder
from pysnn.network import SpikingModule
from pysnn.datasets import OR, Intensity
from pysnn.utils import _set_no_grad


#########################################################
# Params
#########################################################
# Architecture
n_in = 2
n_hidden = 2
n_out = 2

# Data
duration = 100
intensity = 100
num_workers = 0
batch_size = 1

# Neuronal Dynamics
thresh = 1.0
v_rest = 0.0
alpha_v = 1.0
tau_v = 0.7
alpha_t = 1.0
tau_t = 0.7
duration_refrac = 3
dt = 1
delay = 0
i_dynamics = (dt, alpha_t, tau_t)
n_dynamics = (thresh, v_rest, alpha_v, alpha_t, dt, duration_refrac, tau_v, tau_t)
c_dynamics = (batch_size, dt, delay)

# Learning
lr = 0.01


#########################################################
# Network
#########################################################
class Network(SpikingModule):
    def __init__(self):
        super(Network, self).__init__()

        # Input
        self.input = Input((batch_size, 1, n_in), *i_dynamics)

        # layer 1
        self.mlp1_c = Linear(n_in, n_out, *c_dynamics)
        self.readout = GradReadoutNeuron(
            (batch_size, 1, n_out), v_rest, alpha_v, dt, tau_v
        )

    def forward(self, input):
        x, t = self.input(input)

        # Layer 1
        x, t = self.mlp1_c(x, t)
        v = self.readout(x, t)

        return v

    def reset_state(self):
        pass

# class Network(SpikingModule):
#     def __init__(self):
#         super(Network, self).__init__()

#         # Input
#         self.input = Input((batch_size, 1, n_in), *i_dynamics)

#         # layer 1
#         self.mlp1_c = linear(n_in, n_hidden, *c_dynamics)
#         self.neuron1 = gradlifneuron((batch_size, 1, n_hidden), *n_dynamics)

#         # layer 2
#         self.mlp2_c = linear(n_hidden, n_out, *c_dynamics)
#         self.readout = gradreadoutneuron(
#             (batch_size, 1, n_out), v_rest, alpha_v, dt, tau_v
#         )

#     def forward(self, input):
#         x, t = self.input(input)

#         # Layer 1
#         x, t = self.mlp1_c(x, t)
#         x, t = self.neuron1(x, t)

#         # Layer out
#         x, t = self.mlp2_c(x, t)
#         v = self.readout(x, t)

#         return v

#     def reset_state(self):
#         pass


#########################################################
# Dataset
#########################################################
data_transform = transforms.Compose([Intensity(intensity)])

train_dataset = OR(
    data_encoder=PoissonEncoder(duration, dt),
    data_transform=data_transform,
    repeats=int(n_in / 2),
)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
)


#########################################################
# Training
#########################################################
device = torch.device("cpu")
net = Network()
net.mlp1_c.reset_weights("uniform", a=0.3)
# net.mlp2_c.reset_weights("uniform", a=0.5)
w1_start = net.mlp1_c.weight.clone()
# w1_start, w2_start = net.mlp1_c.weight.clone(), net.mlp2_c.weight.clone()
for param in net.parameters():
    param.requires_grad = True

loss_func = nn.CrossEntropyLoss()
learning_rule = SGD(net.parameters(), lr=lr, momentum=0.9)

# Tensorboard
# writer = SummaryWriter()
# sample = next(iter(train_dataloader))[0][..., 0]
# writer.add_graph(net, sample)

# Training loop
sp_n0, tr_n0 = [], []
tr_n1, tr_n2 = [], []
sp_n1, sp_n2 = [], []
v_n1, v_n2 = [], []

for batch in tqdm(train_dataloader):
    sample, label = batch[0], batch[1].squeeze(1)

    # Iterate over input's time dimension
    for idx in range(sample.shape[-1]):
        learning_rule.zero_grad()
        input = sample[..., idx]
        v_out = net(input).squeeze(1)

        loss = loss_func(v_out, label)
        loss.backward(retain_graph=True)
        learning_rule.step()

        sp_n0.append(input.flatten())
        tr_n0.append(net.input.trace.clone().flatten())
        # tr_n1.append(net.neuron1.trace.clone().flatten())
        tr_n2.append(net.readout.trace.clone().flatten())
        # sp_n1.append(net.neuron1.spikes.clone().flatten())
        sp_n2.append(net.readout.spikes.clone().flatten())
        # v_n1.append(net.neuron1.v_cell.clone().flatten())
        v_n2.append(net.readout.v_cell.clone().flatten())

    net.reset_state()

dw1 = (net.mlp1_c.weight - w1_start).detach()
# dw1, dw2 = (net.mlp1_c.weight - w1_start).detach(), (net.mlp2_c.weight - w2_start).detach()
print("Layer 1 weight difference: ", dw1)
# print("Layer 2 weight difference: ", dw2)

# Plotting network state and spikes
_, axes = plt.subplots(nrows=3, ncols=2, sharex="row", sharey="row")
data = map(torch.stack, (sp_n0, tr_n0, tr_n2, sp_n2, v_n2))
# data = map(torch.stack, (sp_n0, tr_n0, tr_n1, tr_n2, sp_n1, sp_n2, v_n1, v_n2))
titles = [
    "spikes input",
    "traces input",
    # "traces L1",
    "traces L2",
    # "spikes L1",
    "spikes L2",
    # "voltage L1",
    "voltage L2",
]
for d, ax, t in zip(data, axes.flatten(), titles):
    d = d.detach()
    for idx in range(d.shape[1]):
        if not "spikes" in t:
            ax.plot(d[:, idx], label=str(idx))
        else:
            plt_data = d[:, idx].nonzero()
            ax.vlines(plt_data, ymin=0, ymax=1, label=str(idx))
        ax.set_title(t)
        ax.legend()

# Plot weight differences before and after training
_, heat_axes = plt.subplots(nrows=1, ncols=2)
heatmap(dw1, ax=heat_axes[0])
heat_axes[0].set_title("Weight differences layer 1")
# heatmap(dw2, ax=heat_axes[1])
# heat_axes[1].set_title("Weight differences layer 2")

plt.show()
