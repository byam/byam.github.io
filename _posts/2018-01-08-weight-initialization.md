---
layout: post
title:  "Weight Initialization"
date:   2018-01-08 18:00:00 +0900
categories: dlnd
fbcomments: true
---

Outline:

- [Scratch](#scratch)
- [Distributions](#distributions)
- [General rule for setting weights](#general-rule-for-setting-weights)

## Scratch
- [Jupyter Notebook](https://github.com/byam/dlnd/blob/master/3.ConvolutionalNeuralNetworks/weight-initialization/weight_initialization.ipynb)

## Distributions

* Uniform Distribution
    - **equal probability** of picking any number from a set of numbers. 
    -  TensorFlow's `tf.random_uniform` function
* Normal Distribution
    - Unlike the **uniform distribution**, the **normal distribution** has a higher likelihood of picking number close to it's **mean**. 
    - TensorFlow's `tf.random_normal` function
* Truncated Normal Distribution
    - The generated values follow a **normal distribution** with specified **mean** and **standard deviation**, 
    - Except that values whose magnitude is more than **2** standard deviations from the **mean** are dropped and re-picked.

## General rule for setting weights

* The **general rule** for setting the **weights** in a neural network is to be close to **zero** without being too small. 
* A good pracitce is to start your **weights** in the range of $$[-y, y]$$ where $$y=1/\sqrt{n}$$ 
    - ($$n$$ is the number of inputs to a given neuron).
