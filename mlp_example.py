import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from pysnn.network import SNNNetwork
from pysnn.connection import LinearExponential
from pysnn.neuron import FedeNeuronTrace
from pysnn.learning import FedeSTDP

from event_pytorch.event_dataloaders import (
    SineData,
    NumpyToTensor,
    DiscretizeFloat,
    RepeatLabels
)
from bindsnet.encoding import PoissonEncoder


#########################################################
# Params
#########################################################
# Architecture
n_in = 5
n_hidden1 = 10
n_hidden2 = 10
n_out = 5

# Data
time = 50
intensity = 128
num_workers = 0
n_samples = 20
batch_size = 20

# Neuronal Dynamics
thresh = 0.8
v_rest = 0
alpha_v = 0.2
tau_v = 5
alpha_t = 1.
tau_t = 5
duration_refrac = 5
dt = 1
delay = 3
n_dynamics = (thresh, v_rest, alpha_v, alpha_v, dt, duration_refrac, tau_v, tau_t)
c_dynamics = (batch_size, dt, delay, tau_t, alpha_t)

# Learning
lr = 0.0001
w_init = 0.5
a = 0.5


#########################################################
# Network
#########################################################
class Network(SNNNetwork):
# class Network(torch.jit.ScriptModule):
    def __init__(self):
        super(Network, self).__init__()

        # Layer 1
        self.mlp1_c = LinearExponential(n_in, n_hidden1, *c_dynamics)
        self.neuron1 = FedeNeuronTrace((batch_size, 1, n_hidden1), *n_dynamics)

        # Layer 2
        self.mlp2_c = LinearExponential(n_hidden1, n_hidden2, *c_dynamics)
        self.neuron2 = FedeNeuronTrace((batch_size, 1, n_hidden2), *n_dynamics)

        # Layer out
        self.mlp3_c = LinearExponential(n_hidden2, n_out, *c_dynamics)
        self.neuron3 = FedeNeuronTrace((batch_size, 1, n_out), *n_dynamics)

        # Learning rule
        connections = [self.mlp1_c, self.mlp2_c, self.mlp3_c]
        self.learning_rule = FedeSTDP(connections, lr, w_init, a)

    # @torch.jit.script_method
    def forward(self, input):
        # Layer 1
        x, t  = self.mlp1_c(input)
        x = self.neuron1(x, t)

        # Layer 2
        x, t = self.mlp2_c(x)
        x = self.neuron2(x, t)

        # Layer out
        x, t = self.mlp3_c(x)
        x = self.neuron3(x, t)

        # Learning
        self.learning_rule()

        return x


#########################################################
# Dataset
#########################################################
data_transform = transforms.Compose([NumpyToTensor(),
                                     transforms.Lambda(lambda x: x*intensity),
                                     DiscretizeFloat()])
label_transform = transforms.Lambda(lambda x: x*intensity)

train_dataset = SineData(M=n_in,
                         n_samples=n_samples,
                         encoder=PoissonEncoder(time=time, dt=dt),
                         label_encoder=PoissonEncoder(time=time, dt=dt),
                         data_transform=data_transform,
                         label_transform=label_transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)


#########################################################
# Training
#########################################################
# device = torch.device("cuda")
# net = Network().cuda()
device = torch.device("cpu")
net = Network()

out = []
for batch in train_dataloader:
    batch = batch["input"].squeeze(2).squeeze(2).to(device)
    for idx in range(batch.shape[1]):
        input = batch[:, idx:idx+1, :]
        out.append(net(input))
    net.reset_state()

output = torch.stack(out, dim=1)
print(output.shape)
