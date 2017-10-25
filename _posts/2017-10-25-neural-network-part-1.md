---
layout: post
title:  "Neural Network Part 1"
date:   2017-10-25 22:35:47 +0900
categories: memo
fbcomments: true
---

## Perceptrons

A perceptron takes several binary inputs, $$x_1,x_2,…$$ and produces a single binary output:

![Perceptrion]({{ "/assets/img/perceptron.png" | absolute_url }}){: .center-image }

*Weights*, $$w_1,w_2,…$$ real numbers expressing the importance of the respective inputs to the output.
The neuron's output, $$0$$ or $$1$$, is determined by whether the weighted sum $$\sum_{j}w_jx_j$$ is less than or greater
 than some threshold value. 

$$
output =
\begin{cases}
0,  if \sum_{j}w_jx_j \le threshold \\[2ex]
1,  if \sum_{j}w_jx_j \gt threshold
\end{cases}
$$
 
Just like the weights, the threshold is a real number which is a parameter of the neuron. 

Using the *bias* instead of the *threshold*, the perceptron rule can be rewritten:

$$
output =
\begin{cases}
0,  if \sum_{j}w_jx_j + bias \le 0 \\[2ex]
1,  if \sum_{j}w_jx_j + bias \gt 0
\end{cases}
$$

Briefly
- **Perceptron** is that it's a device that makes decisions by weighing up evidence. That's the basic mathematical model
- By varying the **weights** and the **threshold**, we can get different models of decision-making.
- **Complex network of perceptrons** could make **quite subtle decisions**.
- We can think of the **bias** as a measure of how easy it is to get the perceptron to output a 1.
- Another way perceptrons can be used is to compute the elementary logical functions 
  we usually think of as underlying computation, functions such as **AND**, **OR**, and **NAND**.
- We can use **networks of perceptrons** to compute **any logical function** at all.

*"We can devise learning algorithms which can automatically tune the weights and biases of a network of artificial neurons. 
This tuning happens in response to external stimuli, without direct intervention by a programmer."*

## Sigmoid Neurons

A small change in the **weights** or **bias** of any single perceptron in the network can sometimes cause 
the output of that perceptron to completely flip, say from **0** to **1**.

![Perceptrion Small Change]({{ "/assets/img/perceptron-small-change.png" | absolute_url }}){: .center-image }

**Sigmoid neurons** are similar to **perceptrons**, but modified so that small changes in their **weights** and 
**bias** cause only a small change in their output.



### References

- [http://neuralnetworksanddeeplearning.com/chap1.html](http://neuralnetworksanddeeplearning.com/chap1.html)
