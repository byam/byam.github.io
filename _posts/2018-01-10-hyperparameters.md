---
layout: post
title:  "Hyperparameters"
date:   2018-01-10 00:00:00 +0900
categories: dlnd
fbcomments: true
---

Outline:

- [Introduction](#introduction)
- [Learning Rate](#learning-rate)
- [Minibatch](#minibatch)

## Introduction

* Hyperparameter
    - **Variable** that we need to **set before** applying a learning algorithm to a data set.
* Challenge
    - There are **no magic numbers** that work everywhere.
    - The best numbers **depend on** each **task** and each **dataset**.

### Hyperparameters categories

#### 1. Optimizer Hyperparameters
- related more the **optimization** and **training process** than to the model itself.
- `learning rate`, the `minibatch size`, and the number of training iterations or `Epochs`.

![hyperparameters-optimizer.png]({{ "/assets/img/rnn/hyperparameters-optimizer.png" | absolute_url }}){: .center-image }{:width="600px"}

#### 2. Model Hyperparameters
- more involved in the **structure of the model**
- the **number of layers** and **hidden units** and **model specific hyperparameters** for architectures like RNNs.

![hyperparameters-model.png]({{ "/assets/img/rnn/hyperparameters-model.png" | absolute_url }}){: .center-image }{:width="600px"}


## Learning Rate

* Good Starting Points
    - then a good starting point is usually `0.01`
    - these are the usual suspects of learning rates $$[0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]$$

* Gradient
    - Calculating the **gradient** would tell us **which direction** to go to **decrease the error**.
    - the gradient will point out which direction to go
* Learning Rate
    - Is the **multiplier** we use to **push the weight** towards the **right direction**.

![learning-rate-scenes.png]({{ "/assets/img/rnn/learning-rate-scenes.png" | absolute_url }}){: .center-image }{:width="600px"}

#### Learning Rate Decay

* It would be **stuck** oscillating between values that still have a better error value than when we started training, but are not the best values possible for the model.
    ![learning-rate-stuck.png]({{ "/assets/img/rnn/learning-rate-stuck.png" | absolute_url }}){: .center-image }{:height="300px"}
* Intuitive ways to do this can be by **decreasing the learning rate linearly**.
    - also decrease the learning rate **exponentially**
    - So, for example we'd multiply the learning rate by 0.1 every 8 epochs for example.
    ![learning-rate-exponent.png]({{ "/assets/img/rnn/learning-rate-exponent.png" | absolute_url }}){: .center-image }{:height="300px"}

#### Adaptive learning rate

* There are more clever learning algorithms that have an adaptive learning rate.
* These algorithms adjust the learning rate **based on what the learning algorithm knows about the problem** and the data that it's seen so far.
    - This means not only **decreasing** the learning rate when needed,
    - but also **increasing** it when it appears to be too low.
* Adaptive Learning Optimizers
  - [AdamOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer)
  - [AdagradOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer)

## Minibatch

* Online(Stochastic) training
    - fit a **single example of data set** to the model during a **training step**
* Batch training
    - the **entire dataset** to the training step

![online-batch.png]({{ "/assets/img/rnn/online-batch.png" | absolute_url }}){: .center-image }{:width="500px"}


* **Minibatch training**
    - **online training** is when the minibatch size is 1
    - **batch training** is when the minibatch size is the same as the number of examples in the training set

![minibatch.png]({{ "/assets/img/rnn/minibatch.png" | absolute_url }}){: .center-image }{:width="400px"}


* Good Starting Points
    - The recommended starting values: $$[1, 2, 4, 8, 16, 32, 64, 128, 256]$$ 
    - `32` often being a good candidate.
    - larger minibatch size
        - allows **computational boos**t that utilizes matrix multiplication in the training calculations
        - but that comes at the expense of **needing more memory** for the training process and generally **more computational resources**.
        - **Some out of memory errors** in Tensorflow can be eliminated by **decreasing** the minibatch size.
    - small minibatch size
        - have more noise in their error calculations and this noise is often helpful in preventing the training process from stopping at local minima on curve
        ![minibatch-small-large.png]({{ "/assets/img/rnn/minibatch-small-large.png" | absolute_url }}){: .center-image }{:width="300px"}


* experimental result
    - too small could be **too slow**,
    - too large could be **computationally taxing** and could result in **worse accuracy**.
    - And **32** to **256** are potentially **good starting values** for you to experiment with.
 
![exp-minibatch-lr.png]({{ "/assets/img/rnn/exp-minibatch-lr.png" | absolute_url }}){: .center-image }{:width="500px"}

![exp-minibatch-lr-change.png]({{ "/assets/img/rnn/exp-minibatch-lr-change.png" | absolute_url }}){: .center-image }{:width="500px"}
