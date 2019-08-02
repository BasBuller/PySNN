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

## Traces
Defining which object keeps track of the traces is up to the user. For ease of use and lower memory footprint the cell can keep track of its
traces. This directly assumes that the traces for every synapse originating from a single neuron are the same. It is possible to keep track
of the traces in the connection object. In this case each synapse can have a separate trace.

## Delay through synapse
Can do the following, have a tensor equal to the number of synapses. For an MLP of 5 pre-synaptic and 10 post-synaptic neurons this matrix
is (10 x 5). Within this matrix store the number of milliseconds left before the spike has been propagated through the synapse. Once the
counter hits 1 generate a spike and decrease to 0. Just make sure to increment ms duration of spike propagation with 1 such that the actual
desired delay is achieved and it is not cut short by 1 due to implementation efficiency. In this implementation the designer of the network
has a responsibility of making sure dt and synaptic delay match well, otherwise counting errors might occur. Also note, the refractory
period of a cell should be equal or longer than the transmission time! If it is not the spike timing might be reset.

## Module definitions
Make sure each module has a reset_state(self) method! It is called from the SNNNetwork class and is needed for proper simulation of multiple
inputs.

## To do
- Adjust learning rule such that it is able to select which weights are learnable and which are not. Also adjust layer class such that the
  parameter __training__ is also used within STDP. Just make sure gradients are always turned off since we don't need those...
- For connection class, make sure it can handle the transmission of multiple spike within the same synapse. Aka, it should be able to handle
  a new incoming spike while the previous one has not passed through the entire synapse or time delay.
- Make sure all indexing operations are replace with matrix multiplications where possible. GPU is considerably better at floating point
  operations than integer and bitwise operations.
- Make both a LinearDelayed and LinearInstant layer
- Move functionalities shared among multiple classes to functional folder.
  decay and trace parameters. Still, batch 4 images is better than no batching.

## Notes
- Is there a way to execute an update cycle for the entire network at once, instead of doing it sequentially like it is done now? Due to the
  delay in a cell state input to a cell is dependent on previous time steps and not on previous layer activations. This should alleviate a
  large part of the sequential computational burden. Just hope the entire network can be loaded onto the GPU at once.