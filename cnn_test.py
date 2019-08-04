from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from snn.network import SNNNetwork
from snn.connection import Conv2dExponential, AdaptiveMaxPool2d
from snn.neuron import FedeNeuronTrace, InputNeuronTrace
from snn.learning import FedeSTDP
from snn.utils import conv2d_output_shape

from event_pytorch.event_dataloaders import NMNISTDataset


#########################################################
# Params
#########################################################
# Architecture
n_in = 5
n_hidden1 = 10
n_hidden2 = 10
n_out = 5

# Data
dataset_path = "/home/basbuller/python_lib_sources/slayerPytorch/example/NMNISTsmall/"
sample_file = "/home/basbuller/python_lib_sources/slayerPytorch/example/NMNISTsmall/train1K.txt"
sample_length = 300
num_workers = 0
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
    def __init__(self):
        super(Network, self).__init__()

        # layer in
        self.input = InputNeuronTrace(n_in)

        # Layer 1
        self.conv1 = Conv2dExponential(2, 4, 5, 34, 34, *c_dynamics, padding=1)
        # conv1_out = conv2d_output_shape(34, 34, 5, padding=1)
        self.neuron1 = FedeNeuronTrace((batch_size, 4, 32, 32), *n_dynamics)

        # Layer 2
        self.pool2 = AdaptiveMaxPool2d((16, 16))
        self.conv2 = Conv2dExponential(4, 8, 5, 16, 16, *c_dynamics, padding=1)
        # conv2_out = conv2d_output_shape(*conv1_out, 3, padding=1)
        self.neuron2 = FedeNeuronTrace((batch_size, 8, 14, 14), *n_dynamics)

        # # Layer out
        self.conv3 = Conv2dExponential(8, 1, 5, 14, 14, *c_dynamics, padding=1)
        # conv3_out = conv2d_output_shape(*conv2_out, 3, padding=1)
        self.neuron3 = FedeNeuronTrace((batch_size, 1, 12, 12), *n_dynamics)

        # # Learning rule
        connections = [self.conv1, self.conv2, self.conv3]
        self.learning_rule = FedeSTDP(connections, lr, w_init, a)

    def forward(self, x):
        # Layer in
        x = self.input(x)

        # Layer 1
        x, t  = self.conv1(x)
        x = self.neuron1(x, t)

        # Layer 2
        x = self.pool2(x) 
        x, t = self.conv2(x)
        x = self.neuron2(x, t)

        # # Layer out
        x, t = self.conv3(x)
        x = self.neuron3(x, t)

        # # Learning
        self.learning_rule()

        return x


#########################################################
# Dataset
#########################################################
train_dataset = NMNISTDataset(dataset_path, sample_file, dt, sample_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


#########################################################
# Training
#########################################################
device = torch.device("cuda")
net = Network()
net = net.to(torch.float16).cuda()

# device = torch.device("cpu")
# net = Network()

for batch in tqdm(train_dataloader):
    input = batch[0]
    for idx in range(input.shape[-1]):
        x = input[:, :, :, :, idx].to(device)
        net(x)