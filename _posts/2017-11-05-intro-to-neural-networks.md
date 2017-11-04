---
layout: post
title:  "Intro to Neural Networks"
date:   2017-11-05 00:15:00 +0900
categories: dlnd
fbcomments: true
---

Outline:
- [The simplest neural network](#the-simplest-neural-network)
- [Sigmoid function](#sigmoid-function)
- [Simple network implementation](#simple-network-implementation)


<iframe width="560" height="315" src="https://www.youtube.com/embed/Mqogpnp1lrU" frameborder="0" allowfullscreen></iframe>{: .center-image }

## The simplest neural network

![simple-neuron]({{ "/assets/img/simple-neuron.png" | absolute_url }}){: .center-image }{:width="400px"}
*Diagram of a simple neural network. Circles are units, boxes are operations*

- The activation function, $$f(h)$$ can be **any function**, not just the *step function* shown earlier.
- Other activation functions are the **logistic** (often called the sigmoid), **tanh**, and **softmax** functions.

## Sigmoid function

![sigmoid]({{ "/assets/img/sigmoid.png" | absolute_url }}){: .center-image }{:width="400px"}
*The Sigmoid Function*

$$ sigmoid(x) = 1 / (1 + e^{-x}) $$

- The sigmoid function is bounded between 0 and 1
- An output can be interpreted as a probability for success.
- It turns out, again, using a sigmoid as the activation function results in the same formulation as logistic regression.


## Simple network implementation

The output of the network is 

$$y = f(h) = sigmoid(\sum_{i}w_ix_i + b)$$

```python
import numpy as np

def sigmoid(x):
    # Implement sigmoid function
    return 1/(1 + np.exp(-x))

inputs = np.array([0.7, -0.3])
weights = np.array([0.1, 0.8])
bias = -0.1

# Calculate the output
output = sigmoid(np.dot(weights, inputs) + bias)

print('Output:')
print(output) # 0.432907095035
```
