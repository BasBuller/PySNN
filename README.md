# PySNN
Spiking neural netowork framework written on top of PyTorch for efficient simulation of SNNs with correlation based methods. The library adheres to the torch.nn.Module design and implements efficient CUDA kernels where needed.
Based on the cuSNN library at https://github.com/tudelft/cuSNN.

## Structure
Mirror the structure of PyTorch framework. Most core functions defined in functional module. Interfaces (classes wrapping functional) are defined in general snn lib.