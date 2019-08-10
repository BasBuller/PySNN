# __PySNN__

Spiking neural netowork framework written on top of PyTorch for efficient simulation of SNNs with correlation based methods. The library adheres to the torch.nn.Module design and implements efficient CUDA kernels where needed.
Based on the cuSNN library at:

    https://github.com/tudelft/cuSNN

## __Structure__

Mirror the structure of PyTorch framework. Most core functions defined in functional module. Interfaces (classes wrapping functional) are
defined in general snn lib.

## __Network Definition__

Important aspect of the package is the decorator for the forward function of the network. This decorator makes it such that the forward pass
progresses over the layers not front to back, but propagates information based on the timesteps. This way the output signal from input 1 has
not reached the end of the network before input 2 has been presented to the network.

## __Connection Shapes__

In order to keep track of traces and delays in information passing tensors an extra dimension is needed compared to the PyTorch conventions. Because of ease of use with Python broadcasting of matrices this extra dimension is added as the 0th dimension, meaning an extra dimension is present before the batch dimension. This hold for trace tensors.

## __Traces__

Defining which object keeps track of the traces is up to the user. For ease of use and lower memory footprint the cell can keep track of its
traces. This directly assumes that the traces for every synapse originating from a single neuron are the same. It is possible to keep track
of the traces in the connection object. In this case each synapse can have a separate trace.

## __Delay through synapse__

Can do the following, have a tensor equal to the number of synapses. For an MLP of 5 pre-synaptic and 10 post-synaptic neurons this matrix
is (10 x 5). Within this matrix store the ninformation passing tensorsr of milliseconds left before the spike has been propagated through the synapse. Once the
counter hits 1 generate a spike and decreasinformation passing tensors 0. Just make sure to increment ms duration of spike propagation with 1 such that the actual
desired delay is achieved and it is not cutinformation passing tensorsrt by 1 due to implementation efficiency. In this implementation the designer of the network
has a responsibility of making sure dt and synaptic delay match well, otherwise counting errors might occur. Also note, the refractory
period of a cell should be equal or longer than the transmission time! If it is not the spike timing might be reset.

## __Module definitions__

Make sure each module has a reset_state(self) method! It is called from the SNNNetwork class and is needed for proper simulation of multiple
inputs.

## __To do__

- Adjust learning rule such that it is able to select which weights are learnable and which are not. Also adjust layer class such that the
  parameter __training__ is also used within STDP. Just make sure gradients are always turned off since we don't need those...
- For connection class, make sure it can handle the transmission of multiple spike within the same synapse. Aka, it should be able to handle
  a new incoming spike while the previous one has not passed through the entire synapse or time delay.
- Make sure all indexing operations are replace with matrix multiplications where possible. GPU is considerably better at floating point
  operations than integer and bitwise operations.
- Make both a LinearDelayed and LinearInstant layer
- Move functionalities shared among multiple classes to functional folder.
- Decide on where to check for refractory cells, it might be that the voltage update function is not the best place?
