import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from pysnn.connection import Linear
from pysnn.neuron import FedeNeuronTrace, InputTraceExponential
from pysnn.learning import FedeSTDP
from pysnn.encoding import PoissonEncoder
from pysnn.network import SNNNetwork
from pysnn.datasets import OR, BooleanNoise


#########################################################
# Params
#########################################################
# Architecture
n_in = 6
n_hidden = 10
n_out = 1

# Data
duration = 200
intensity = 40
num_workers = 0
batch_size = 1

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
i_dynamics = (dt, alpha_t, tau_t)
n_dynamics = (thresh, v_rest, alpha_v, alpha_v, dt, duration_refrac, tau_v, tau_t)
c_dynamics = (batch_size, dt, delay)

# Learning
lr = 0.0001
w_init = 0.5
a = 0.5


#########################################################
# Network
#########################################################
class Network(SNNNetwork):
    def __init__(self):
        super(Network, self).__init__()

        # Input
        self.input = InputTraceExponential((n_in,), *i_dynamics)

        # Layer 1
        self.mlp1_c = Linear(n_in, n_hidden, *c_dynamics)
        self.neuron1 = FedeNeuronTrace((batch_size, 1, n_hidden), *n_dynamics)

        # Layer 2
        self.mlp2_c = Linear(n_hidden, n_out, *c_dynamics)
        self.neuron2 = FedeNeuronTrace((batch_size, 1, n_out), *n_dynamics)

        # # Learning rule
        # connections = [self.mlp1_c, self.mlp2_c]
        # self.learning_rule = FedeSTDP(connections, lr, w_init, a)

    # @torch.jit.script_method
    def forward(self, input):
        x, t = self.input(input)

        # Layer 1
        x, t  = self.mlp1_c(x, t)
        x, t = self.neuron1(x, t)

        # Layer out
        x, t = self.mlp2_c(x, t)
        x, t = self.neuron2(x, t)

        # Learning
        self.learning_rule()

        return x


#########################################################
# Dataset
#########################################################
data_transform = BooleanNoise(0.1, 0.9)
lbl_transform = transforms.Lambda(lambda x: x*intensity)

train_dataset = OR(data_encoder=PoissonEncoder(duration, dt),
                   data_transform=data_transform,
                   lbl_transform=lbl_transform,
                   repeats=3)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=num_workers)


#########################################################
# Training
#########################################################
device = torch.device("cpu")
net = Network()

out = []
for batch in tqdm(train_dataloader):
    sample, label = batch
    for idx in range(sample.shape[-1]):
        input = sample[:, :, idx].unsqueeze(1)
        out.append(net(input))
    net.reset_state()

output = torch.stack(out, dim=1)
print(output.shape)
