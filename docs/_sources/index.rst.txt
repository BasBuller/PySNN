.. PySNN documentation master file, created by
   sphinx-quickstart on Sun Oct 27 14:48:11 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PySNN's documentation!
=================================

PySNN is a spiking neural network (SNN) framework written on top of PyTorch for efficient simulation of SNNs both on CPU and GPU. The framework is intended for with correlation based learning methods. The library adheres to the highly modular and dynamic design of PyTorch, and does not require its user to learn a new framework like when using BindsNet.

This framework's power lies in the ease of defining and mixing new Neuron and Connection objects that seamlessly work together, even different versions, in a single network.

PySNN is designed to mostly provide low level objects to its user that can be combined and mixed. The biggest difference with PyTorch is that a network now consists of two types of modules, instead of the single nn.Module in regular PyTorch. These new modules are the pysnn.Neuron and pysnn.Connection. 

.. toctree::
   :maxdepth: 2
   :caption: Usage:

   installation
   quickstart
   neurons
   connections
   learning_rules
   networks

.. toctree::
   :maxdepth: 2
   :caption: Package Reference:

   connection_reference
   neuron_reference
   network_reference
   file_io_reference
   functional_reference
   encoding_reference
   datasets_reference
   learning_reference
   utils_reference


Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
.. * :ref:`modindex`
