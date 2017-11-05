---
layout: post
title:  "Gradient Descent"
date:   2017-11-05 14:50:00 +0900
categories: dlnd
fbcomments: true
---

Outline:
- [Learning weights](#learning-weights)
- [Gradient Descent](#gradient-descent)
- [Gradient Descent Math](#gradient-descent-math)
- [Gradient Descent Code](#gradient-descent-code)
- [Gradient Descent Implementation](#gradient-descent-implementation)
- [Reference](#reference)

## Learning weights

- We'll need to **learn the weights** from example data, then use those weights to make the predictions.
- We want the network to make predictions as close as possible to the real values.
- To measure this, we need a metric of how wrong the predictions are, **the error**. 

![network]({{ "/assets/img/gradient/network.png" | absolute_url }}){: .center-image }{:width="600px"}

#### Error function
A common metric is the **sum of the squared errors (SSE)**:

![sse]({{ "/assets/img/gradient/sse.png" | absolute_url }}){: .center-image }{:width="500px"}


- We want the network's prediction error to be as small as possible and the **weights are the knobs** we can use to make that happen. 
- Our goal is to find weights $$w​_{ij}$$ that minimize the squared error $$E$$. 
- To do this with a neural network, typically you'd use **gradient descent**.

## Gradient Descent

- With gradient descent, we take **multiple small steps** towards our goal.
- We want to **change the weights** in steps that **reduce the error**.
- The gradient is just a **derivative** generalized to functions with more than one variable.
- We can use calculus to find the gradient at any point in our error function, which depends on the **input weights**.

![gradient]({{ "/assets/img/gradient/gradient.png" | absolute_url }}){: .center-image }{:width="500px"}

#### Derivative example

- The derivative of $$x^{​2}$$ is $$f^{\prime}(x)=2x$$.
- So, at $$x=2$$, **the slope** is $$f^{\prime}(2)=4$$. 
- Plotting this out, it looks like:

![derivative-example]({{ "/assets/img/gradient/derivative-example.png" | absolute_url }}){: .center-image }{:width="500px"}


#### Contour Map

- Example of the error of a neural network with **two inputs**, and accordingly, **two weights**. 
- At each step, you calculate **the error** and **the gradient**.
- Then use those to determine **how much to change each weight**. 
- Repeating this process will eventually **find weights** that are close to **the minimum of the error function**, the block dot in the middle.

![gradient-descent]({{ "/assets/img/gradient/gradient-descent.png" | absolute_url }}){: .center-image }{:width="500px"}

## Gradient Descent Math

#### Learning Rate

![learning-rate]({{ "/assets/img/gradient/learning-rate.png" | absolute_url }}){: .center-image }{:width="500px"}

#### Chain Rule

![chain-rule]({{ "/assets/img/gradient/chain-rule.png" | absolute_url }}){: .center-image }{:width="500px"}

#### Derivative

![calculation-1]({{ "/assets/img/gradient/calculation-1.png" | absolute_url }}){: .center-image }{:width="500px"}

![calculation-2]({{ "/assets/img/gradient/calculation-2.png" | absolute_url }}){: .center-image }{:width="500px"}

![calculation-3]({{ "/assets/img/gradient/calculation-3.png" | absolute_url }}){: .center-image }{:width="500px"}

#### Define error term

![calculation-4]({{ "/assets/img/gradient/calculation-4.png" | absolute_url }}){: .center-image }{:width="500px"}

#### Multiple Outpus

![calculation-5]({{ "/assets/img/gradient/calculation-5.png" | absolute_url }}){: .center-image }{:width="500px"}

![calculation-6]({{ "/assets/img/gradient/calculation-6.png" | absolute_url }}){: .center-image }{:width="500px"}


## Gradient Descent Code

- This code is for the case of **only one output unit**. 
- We'll also be using the **sigmoid** as the activation function 

```python
import numpy as np

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    """
    # Derivative of the sigmoid function
    """
    return sigmoid(x) * (1 - sigmoid(x))

learnrate = 0.5
x = np.array([1, 2, 3, 4])
y = np.array(0.5)

# Initial weights
w = np.array([0.5, -0.5, 0.3, 0.1])

### Calculate one gradient descent step for each weight
### Note: Some steps have been consilated, so there are
###       fewer variable names than in the above sample code

# TODO: Calculate the node's linear combination of inputs and weights
h = np.dot(x, w)

# TODO: Calculate output of neural network
nn_output = sigmoid(h)

# TODO: Calculate error of neural network
error = y - nn_output

# TODO: Calculate the error term
#       Remember, this requires the output gradient, which we haven't
#       specifically added a variable for.
error_term = error * sigmoid_prime(h)

# TODO: Calculate change in weights
del_w = learnrate * error_term * x

print('Neural Network output:')
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)
```

## Gradient Descent Implementation



## Reference

- [Gradient tutorial of Khan Academy](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/gradient-and-directional-derivatives/v/gradient)
