# __PySNN__

Spiking neural network framework written on top of PyTorch for efficient simulation of SNNs with correlation based methods. The library adheres to the torch.nn.Module design.
Based on the cuSNN library at:

    https://github.com/tudelft/cuSNN

## __Installation__

Installation can be done with pip:

```bash
$ git clone https://github.com/BasBuller/PySNN
$ pip install -e PySNN/
```
Code is formatted with [Black](https://github.com/psf/black) using a pre-commit hook. To configure it, run:

```bash
$ pre-commit install
```

## __Structure__

Mirror the structure of PyTorch framework. Most core functions defined in functional module. Interfaces (classes wrapping functional) are defined in general snn lib.

## __Network Definition__

The overall structure of a network definition is the same as in PyTorch where possible. The biggest differences are as follows:

- Each layer requires both a Connection and a Neuron object as they both implement specific time based dynamics.
- The network definition currently also includes the value/loss function and learning rule definitions. Looking to change this to be same as
  the PyTorch API.

## __Connection Shapes__

In order to keep track of traces and delays in information passing tensors an extra dimension is needed compared to the PyTorch conventions. 
Due to the addition of spike traces, each spiking tensor contains an extra trace dimension as the last dimension. The resulting dimension ordering is as follows for an image tensor (trace is indicated as R to not be confused with time for video data):

    [batch size, channels, height, width, traces] (B,C,H,W,R)

For fully connected layers the resulting tensor is as follows (free dimension can be used the same as in PyTorch):

    [batch size, free dimension, input elements, traces] (B,F,I,R)

Currently, no explicit 3D convolution is possible like is common within video-processing. Luckily, SNNs have a temporal dimension inherently and are (currently still theoretically) well suited for processing videos event by event, and thus not needing 3D convolution.

## __Traces__

Traces are stored both in the Neuron and Connection objects. Currently, Connection objects takes traces from their pre-synaptic Neurons and propagate the trace over time, meaning it does not do any further processing on the traces. If it is desired, one can implement separate trace processing in a custom Connection object.

Traces are stored in a tensor in each Connection, as well as the delay for each trace propagating through the Connection. Only one trace (or signal) can tracked through each synapse. In case delay times through a synapse become very long (longer than the refractory period of the pre-synaptic cell) it is possible for a new signal to enter the Connection before the previous one has travelled through it. In the current implementation the old signal will be overwritten, meaning the information is lost before it is used!

    It is up to the user to assure refractory periods are just as long or longer than the synaptic delay in the following Connection!

## __Module definitions__

Make sure each module has a reset_state(self) method! It is called from the SNNNetwork class and is needed for proper simulation of multiple
inputs.

## __To do__

- Allow for having a local copy of a cell's entire trace history. Possibly also extending this to Connection objects. This will result in a large increase in memory usage.
- Change from using .uint8 to .bool datatypes with the introduction of PyTorch 1.2.

### __Learning rules__

- Adjust learning rule such that it is able to select which weights are learnable and which are not. 
- Adjust layer class such that the parameter __training__ is also used within STDP. Just make sure gradients are always turned off since we don't need those...

### __Connection classes__

- For connection class, make sure it can handle the transmission of multiple spike within the same synapse. Aka, it should be able to handle
