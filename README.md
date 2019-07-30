# PySNN
Spiking neural netowork framework written on top of PyTorch for efficient simulation of SNNs with correlation based methods. The library adheres to the torch.nn.Module design and implements efficient CUDA kernels where needed.
Based on the cuSNN library at https://github.com/tudelft/cuSNN.

## Structure
Mirror the structure of PyTorch framework. Most core functions defined in functional module. Interfaces (classes wrapping functional) are
defined in general snn lib.

## Network Definition
Important aspect of the package is the decorator for the forward function of the network. This decorator makes it such that the forward pass
progresses over the layers not front to back, but propagates information based on the timesteps. This way the output signal from input 1 has
not reached the end of the network before input 2 has been presented to the network.

## Delay through synapse
Can do the following, have a tensor equal to the number of synapses. For an MLP of 5 pre-synaptic and 10 post-synaptic neurons this matrix
is (10 x 5). Within this matrix store the number of milliseconds left before the spike has been propagated through the synapse. Once the
counter hits 1 generate a spike and decrease to 0. Just make sure to increment ms duration of spike propagation with 1 such that the actual
desired delay is achieved and it is not cut short by 1 due to implementation efficiency. In this implementation the designer of the network
has a responsibility of making sure dt and synaptic delay match well, otherwise counting errors might occur. Also note, the refractory
period of a cell should be equal or longer than the transmission time! If it is not the spike timing might be reset.


## To do
- For connection class, make sure it can handle the transmission of multiple spike within the same synapse. Aka, it should be able to handle
  a new incoming spike while the previous one has not passed through the entire synapse or time delay.