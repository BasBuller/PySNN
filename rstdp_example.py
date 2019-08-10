from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from pysnn.network import SNNNetwork
from pysnn.connection import Conv2dExponential, AdaptiveMaxPool2d, LinearExponential
from pysnn.neuron import FedeNeuronTrace
from pysnn.learning import RSTDPRateSingleElement, FedeSTDP
from pysnn.utils import conv2d_output_shape

from event_pytorch.event_dataloaders import NMNISTDataset


#########################################################
# Params
#########################################################
# Data
dataset_path = "/home/basbuller/python_lib_sources/slayerPytorch/example/NMNISTsmall/"
sample_file = "/home/basbuller/python_lib_sources/slayerPytorch/example/NMNISTsmall/train1K.txt"
test_file = "/home/basbuller/python_lib_sources/slayerPytorch/example/NMNISTsmall/test100.txt"
sample_length = 300
num_workers = 0
batch_size = 20

# Time and trace decay
dt = 1
alpha_t = 1.
tau_t = 5

# Neuronal dynamics
thresh = 0.5
v_rest = 0.
alpha_v = 0.3
tau_v = 5
duration_refrac = 5
n_dynamics = (thresh, v_rest, alpha_v, alpha_t, dt, duration_refrac, tau_v, tau_t)

# Connection dynamics
delay = 1
c_dynamics = (batch_size, dt, delay, tau_t, alpha_t)

# Learning
epochs = 50
lr = 0.0001
w_init = 0.5
a = 0.5
count_start = 1


#########################################################
# Network
#########################################################
class Network(SNNNetwork):
    def __init__(self):
        super(Network, self).__init__()

        # Layer 1
        self.conv1 = Conv2dExponential(2, 4, 5, (34, 34), *c_dynamics, padding=1)
        # conv1_out = conv2d_output_shape(34, 34, 5, padding=1, stride=2)
        self.neuron1 = FedeNeuronTrace((batch_size, 4, 32, 32), *n_dynamics)

        # Layer 2
        self.pool2 = AdaptiveMaxPool2d((16, 16))
        self.conv2 = Conv2dExponential(4, 8, 3, (16, 16), *c_dynamics, padding=1)
        # conv2_out = conv2d_output_shape(*conv1_out, 3, padding=1)
        self.neuron2 = FedeNeuronTrace((batch_size, 8, 16, 16), *n_dynamics)

        # Layer out
        self.pool3 = AdaptiveMaxPool2d((8, 8))
        self.conv3 = Conv2dExponential(8, 16, 3, (8, 8), *c_dynamics, padding=1)
        # conv3_out = conv2d_output_shape(*conv2_out, 3, padding=1)
        self.neuron3 = FedeNeuronTrace((batch_size, 16, 8, 8), *n_dynamics)

        # Linear
        self.linear4 = LinearExponential(1024, 10, batch_size, dt, delay, tau_t, alpha_t)
        self.neuron4 = FedeNeuronTrace((batch_size, 1, 10), *n_dynamics)

        # Learning rule
        conv_connections = [self.conv1, self.conv2, self.conv3]
        self.unsupervised = FedeSTDP(conv_connections, lr, w_init, a)
        self.supervised = RSTDPRateSingleElement(self.linear4, lr, w_init, a, (batch_size, 10), count_start)

    def forward(self, x):
        # Layer 1
        x, t  = self.conv1(x)
        x = self.neuron1(x, t)

        # Layer 2
        x = self.pool2(x)
        x, t = self.conv2(x)
        x = self.neuron2(x, t)

        # Layer out
        x = self.pool3(x)
        x, t = self.conv3(x)
        x = self.neuron3(x, t)

        # Linear
        x = x.view(x.shape[0], 1, -1)
        x, t = self.linear4(x)
        x = self.neuron4(x, t)

        return x

    def update_weights(self, x, label):
        self.unsupervised()
        self.supervised(x.squeeze(1), label)


#########################################################
# Dataset
#########################################################
train_dataset = NMNISTDataset(dataset_path, sample_file, dt, sample_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

test_dataset = NMNISTDataset(dataset_path, test_file, dt, sample_length)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


#########################################################
# Training
#########################################################
device = torch.device("cuda")
net = Network()
net = net.to(torch.float16).cuda()

# device = torch.device("cpu")
# net = Network()

for epoch in range(epochs):
    print("######################## Epoch {} ########################".format(epoch))
    # Train
    for batch in tqdm(train_dataloader):
        input = batch[0]
        label = batch[2]
        for idx in range(input.shape[-1]):
            x = input[:, :, :, :, idx].to(device)
            label = label.to(device)
            x = net(x)
            net.update_weights(x, label)
        net.reset_state()

    # Test
    correct = 0
    for batch in tqdm(test_dataloader):
        input = batch[0]
        label = batch[2]

        single_label = 0
        for idx in range(input.shape[-1]):
            x = input[:, :, :, :, idx].to(device)
            label = label.to(device)
            x = net(x)
            single_label += x

        _, max_idx = single_label.squeeze(1).max(-1)
        cor_out = max_idx == label
        correct += cor_out.float().sum().cpu().numpy()

        net.reset_state()
    print("{}/{} correct outputs".format(correct, len(test_dataset)))
    print("\n")
