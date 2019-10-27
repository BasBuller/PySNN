pysnn\.connection
=================

A :class:`Connection` object is used to pass spikes, activations, and traces between two sets of :class:`BaseNeuron` objects. The :class:`Connection` is the base
class that should be inherited by each custom connection. Each :class:`Connection` shares a set of basic functionalities/dynamics:

* **Presynaptic Neuron**, which is the :class:`Neuron` that precedes the :class:`Connection`.
* **Postsynaptic Neuron**, which is the :class:`Neuron` that succedes the :class:`Connection`.
* **Delay** in the transmission of incoming signals from its presynaptic :class:`Neuron` to its postsynaptic :class:`Neuron`.
* **Weights**, which are the relative weights that are assigned from spikes originating from each presynaptic :class:`Neuron`.
* **Signals** from the presynaptic :class:`Neuron` that are passed on by the :class:`Connection`, which are the following:
    * Traces
    * Spikes
    * Activation potential, this is not actually an input but a result of multiplying incoming spikes with the connection weight.

Connections
-----------

.. automodule:: pysnn.connection
