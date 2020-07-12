# PySNN Notes MB

Personal development notes.

## Tracing computational graph

The following options exist for tracing the computational graph:
    1. __Winner__ Add a *_children* field to the tensors that pass through an object and call. This can be used to construct the graph.
    2. Send a tracer object through the network. This trace object is simply a PyTorch tensor with extended functionalities.
    3. Somehow make use of PyTorch its autograd capabilities. Possibly by making neurons and connection autograd functions?

### Consequences for choosing option 1

So option 1 it is. This makes it that the following conditions apply:
    * Input knows no _prev node.
    * Lack of _prev node means no Input object is used, which is fine.

### Implementing graph tracing using forward hooks

A tensor _carries_ information about the SpikingModule that created it. When it arrives at the next SpikingModule, this information is
stored in the new SpikingModule. Once the SpikingModule generates output tensors it tags them with __self__, and so the chain continues.
--> This require application of hooks ONLY to the SpikingModules, and not all other intermediate modules.

### Layer = neur - conn - neur

Only interested in layer = (presyn neuron, connection, postsyn neuron), should eliminate (connection, neuron, connection). Options to obtain the desired result:
    1. Generate topology by taking 2 steps at a time.
        * Does require _starting with neuron_.
        * How do I deal with _feedback connections_? I think this is no problem at all. Neurons are added to the past nodes set if the neuron is a postsynaptic neuron. This does not exclude past postsynaptic neurons from being used as a presynaptic neuron in other layers.
    2. Generate (node, edge) objects and combine into layers afterwards.
        * Seems inefficient and tricky.

__Shortcoming of implementation__: The current algorithm fails in situations where two modules are placed between a presyn and postsyn neuron. For example, a connection followed by a pooling operation. If both inherit from SpikingModule, the graph tracing will fail. How to account for this?
    * Type checking for neurons?
    * If using type checking, can recurse over connections till a neuron object is a reached. This might actually be a nice solution.
    * Make the graph dynamic??? Since _self._past_ attribute is a Python object, we can adjust its next element! This way we can dynamically make the graph double and directed. In turn, this allows for tracing the graph from root to leaves, and vice versa.

### Base class learning rule

Requirements:
    * Update function applied to each layer.
    * Modulation of weight updates based on global performance.
    * Group based modulation based on any metric. For example, all recurrent connections are modulated differently that feedforward ones. --> Grouping of layers, each can be modulated differently.

## Gradient based optimization using PySNN

Is it possible to exclude certain operations from the gradient graph? i.e. voltage update operations.
    * Use _hooks_ to exclude operations from gradient graph, i.e., use hooks for changing neuron state :).
    * Identify no-grad ops with a _decorator_ that places it in pre_hook or post_hook category.
    * The forward pass simply uses a _autograd spiking function_.

### Grad TODO

    * [ ] Can grads work with non-grad ops, i.e. buffers?
    * [ ] What is the best interface for voltage update?
    * [ ] What is the best interface for spiking function?
    * [ ] Are there other operations that need a gradient to be tracked/defined?

## TODO

[ ] Implement flexible gradient-based learning.
[ ] Make correlation-based learning work with multiple connections between neurons.
[x] Implement graph tracing.
[x] Layers in a graph contain:
    - Presynaptic neuron
    - Connection
    - Postsynaptic neuron
[x] Make base learning rule compatible with layers format.
[x] Implement *change_batch_size* in BaseNeuron, BaseInput, and Connection objects.
[x] Implement automatic recursion on reset_state function. The user _should not_ have to call super.reset_state() after redifing reset state. Use function *reset_network_state* instead.
[ ] Adapt neurons (and connections?) to allow for multiple inputs.
