# ScalarGrad
A small scalar-valued automatic differentiation engine capable of fundamental arithmetic operations , handling Activation Functions such as tanh, ReLU and sigmoid
and computes gradient via backpropagation.
It is paired with a basic neural network library which is flexible in architecture.

## Example Usage
```
from engine import Value
from compute_graph import *
n = Neuron(2)
x = [Value(1.0),Value(5.0)]
y = n(x)
dot = draw_dot(y)
```
To efficiently track the gradients , compute_graph.py produces a DAG similar to the approach in PyTorch.
!![TrackGradient](https://github.com/user-attachments/assets/a9f3152b-e9d6-4a05-85cd-98c4ebb9feec)

