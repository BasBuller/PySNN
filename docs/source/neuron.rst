pysnn\.neuron
=============

The :class:`Neuron` is the basic and most fundamental object that is used in constructing a spiking neural network (SNN). Each new neuron
design should inherit from the :class:`BaseNeuron` class. Each :class:`Neuron` shares a set of basic functionalities/dynamics:

* **Internal (possibly decaying) voltage** that represents recent incoming activity.
* **Spiking mechanism**, once the voltage of the neuron surpasses its threshold value it will generate a :class:`Boolean` spike.
* **Refractory period/mechanism** that is activated once a :class:`Neuron` has spiked. During this period the :class:`Neuron` is incapable, or less likely, to spike again.
* **Trace** that is a numerical representation of recent activity of the :class:`Neuron`.


Neurons
-------

.. automodule:: pysnn.neuron
