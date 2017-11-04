---
layout: post
title:  "Neural Network Part 1"
date:   2017-10-25 22:35:47 +0900
categories: memo
fbcomments: true
---

- [Perceptrons](#perceptrons)
- [Sigmoid Neurons](#sigmoid-neurons)
- [The architecture of neural networks](#the-architecture-of-neural-networks)

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

## The architecture of neural networks

*Neural network* нь дараах ерөнхийдөө дараах 3 давхаргаас бүрддэг.

- ***Input Layer*** буюу оролтын давхарга
    - Хамгийн зүүн захын давхаргыг хэлэх ба *input neurons* -уудыг агуулж байдаг
- ***Output Layer*** буюу гаралтын давхарга
    - Хамгийн баруун захын давхаргыг хэлэх ба *output neurons* -уудыг агуулж байдаг
- ***Hidden Layer*** буюу нуугдал давхарга
    - *input layer* болон *output layer* -ээс бусад давхаргыг хэлнэ.

Жишээ нь доорх *neural network* нь 4-н давхаргаас бүтсэн бөгөөд, 2 *hidder layer* агуулсан байна.

![The architecture of neural networks]({{ "/assets/img/The architecture of neural networks.png" | absolute_url }}){: .center-image }

*Input layer* болон *output layer* -ийн хэлбэрийг дүрслэх нь хялбархан байх ба 
эсрэгээрээ *hidden layer*-ийн хэлбэрийг дүрслэх нь нарийн тооцоолол шаардана.

Жишээ нь гараар бичсэн `0-9` хүртэл бичсэн цифрүүдийг таниулах зориулалттай *neural network* зохиох үед, 
хэрэв оролтын зураг нь `64x64` гэсэн хэмжээтэй бол *input layer* нь 4096 *neuron*-аас бүрдэх ба,
*output layer* нь `0-9` хүртэлх буюу 10 *neurons* -аас бүрдэх байдлаар хялбархан зохиож болно.

Тэгвэл *hidden layer* -ийг ямар хэлбэртэй байхыг тооцоолох боломжгүй боловч судлаачид 
өөрсдийн гэсэн **heuristics** хэлбэрийг гаргасан байдаг.

Нэг давхаргын гаралт нь дараагийн давхаргын оролт болсон *neural network*-ийг **feadforward** *neural network* гэж нэрлэнэ.
Өөрөөр хэлбэл *feedforward neural network*-т *loop* буюу тойрог байх боломжгүй юм.

Гэсэн хэдий ч **feedback** төрлийн, өөрөөсөө өмнөх давхаргатаа нөлөөлдөг *neural network* байдаг ба түүнийг
 **Recurrent Neural Network (RNN)** гэж нэрлэдэг.

### References

- [http://neuralnetworksanddeeplearning.com/chap1.html](http://neuralnetworksanddeeplearning.com/chap1.html)
