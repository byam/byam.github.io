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
- [Number of Training Iterations](#number-of-training-iterations)
- [Number of Hidden Units Layers](#number-of-hidden-units-layers)
- [RNN Hyperparameters](#rnn-hyperparameters)
- [Sources and References](#sources-and-references)

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


## Number of Training Iterations

* To choose the right number of iterations or number of epochs for our training step,
    - the metric we should have our eyes on is the **validation error**.
* Early Stopping
    - determine when to **stop** training a model
    - roughly works by **monitoring the validation error** and stopping the training when it stops decreasing.

![epochs.png]({{ "/assets/img/rnn/epochs.png" | absolute_url }}){: .center-image }{:width="300px"}

## Number of Hidden Units Layers

![hidden-simple-complex.png]({{ "/assets/img/rnn/hidden-simple-complex.png" | absolute_url }}){: .center-image }{:width="500px"}

* The number and architecture of the **hidden units** is the main measure for a model's learning **capacity**.
    - Provide the model with **too much capacity**
        - it might tend to **overfit** and just try to **memorize** the training set.
        - meaning that the **training accuracy** is much better than the **validation accuracy**
        ![hidden-accuracy.png]({{ "/assets/img/rnn/hidden-accuracy.png" | absolute_url }}){: .center-image }{:width="500px"}
    - Might want to try to **decrease the number of hidden units**
    - Utilize regularization techniques like dropouts or L2 regularization
    ![hidden-utilize.png]({{ "/assets/img/rnn/hidden-utilize.png" | absolute_url }}){: .center-image }{:width="500px"}

 
#### The number of hidden units
1. **The more** hidden units is the better
    - a **little larger** than the ideal number is **not a problem**
    -  **much larger** value can often lead to the model **overfitting** 
2. **If model is not training**
    - **add more hidden units** and **track validation error**
    - **keep adding hidden units** until the **validation starts getting worse**.
3. Another **heuristic** involving the **first hidden layer**
    - **larger than** the number of the **inputs** has been observed to be beneficial in a number of tests

#### The number of layers
1. It's often the case that a **three-layer** neural net will **outperform** a **two-layer** net
    - but going even **deeper** is **rarely helps** much more.
2. The **exception**
    - **Convolutional** neural networks where the deeper they are, the better they perform.

## RNN Hyperparameters

* Two main choices we need to make when we want to build RNN
    1. choosing cell type
        - long short-term memory cell
        - vanilla RNN cell
        - gated recurrent unit cell
    2. how deep the model is
    ![rnn-hype-layers.png]({{ "/assets/img/rnn/rnn-hype-layers.png" | absolute_url }}){: .center-image }{:width="500px"}

* In practice, LSTMs and GRUs perform better than vanilla RNNs
    - While **LSTM**s seem to be more **commonly** used
    ![rnn-hype-cell.png]({{ "/assets/img/rnn/rnn-hype-cell.png" | absolute_url }}){: .center-image }{:width="500px"}
    - It really depends on the **task** and the **dataset**.

## Sources and References

If you want to learn more about hyperparameters, these are some great resources on the topic:

- [Practical recommendations for gradient-based training of deep architectures](https://arxiv.org/abs/1206.5533) by Yoshua Bengio
- [Deep Learning book - chapter 11.4: Selecting Hyperparameters](http://www.deeplearningbook.org/contents/guidelines.html) by Ian Goodfellow, Yoshua Bengio, Aaron Courville
- [Neural Networks and Deep Learning book - Chapter 3: How to choose a neural network's hyper-parameters?](http://neuralnetworksanddeeplearning.com/chap3.html#how_to_choose_a_neural_network's_hyper-parameters) by Michael Nielsen
- [Efficient BackProp (pdf)](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) by Yann LeCun

More specialized sources:

- [How to Generate a Good Word Embedding?](https://arxiv.org/abs/1507.05523) by Siwei Lai, Kang Liu, Liheng Xu, Jun Zhao
- [Systematic evaluation of CNN advances on the ImageNet](https://arxiv.org/abs/1606.02228) by Dmytro Mishkin, Nikolay Sergievskiy, Jiri Matas
- [Visualizing and Understanding Recurrent Networks](https://arxiv.org/abs/1506.02078) by Andrej Karpathy, Justin Johnson, Li Fei-Fei
