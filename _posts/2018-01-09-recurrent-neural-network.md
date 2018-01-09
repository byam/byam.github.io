---
layout: post
title:  "Recurrent Neural Network"
date:   2018-01-09 09:00:00 +0900
categories: dlnd
fbcomments: true
---

Outline:

- [Intro to RNN](#intro-to-rnn)
- [LSTM](#lstm)
- [Character-wise RNN](#character-wise-rnn)
- [Resources](#resources)


## Intro to RNN

* In **feed forward** networks, there is **no sense of order** in inputs.
* Idea is that, **build order** in network. (include information about order)
    - **split data** into parts (text -> words)
    - routing the hidden layer output from the **previous step back** into **hidden layer**
* This architecture colled **Recurrent Neural Network**(RNN).
    - total input of hidden layer is sum of the combinations from **input layer** and **previous hidden layer**. 
    ![steep-rnn.png]({{ "/assets/img/rnn/steep-rnn.png" | absolute_url }}){: .center-image }{:heights="500px"}

* Example
    - word -> characters. (steep -> 's', 't', 'e', 'e', 'p')
    ![steep-example.png]({{ "/assets/img/rnn/steep-example.png" | absolute_url }}){: .center-image }{:heights="500px"}
    ![steep-example-num.png]({{ "/assets/img/rnn/steep-example-num.png" | absolute_url }}){: .center-image }{:heights="500px"}

## LSTM

![hidden-multiply.png]({{ "/assets/img/rnn/hidden-multiply.png" | absolute_url }}){: .center-image }{:heights="100px"}
* In RNN, hidden layer multiplication leads to problem, **gradient** going to
    - really small and **vanish**
    - really large and **explode**
    ![vanishing-exploding.png]({{ "/assets/img/rnn/vanishing-exploding.png" | absolute_url }}){: .center-image }{:heights="500px"}
* We can think of RNN as
    - **bunch of cells** with **inputs** and **outputs**
    - **inside** the cells there are **network layers**
    ![rnn-cell.png]({{ "/assets/img/rnn/rnn-cell.png" | absolute_url }}){: .center-image }{:heights="500px"}    
* To solve the problem of **vanishing gradients**
    - Use more complicated cells, called `LSTM` (Long Short Term Memory)

### LSTM cell

![lstm-cell.png]({{ "/assets/img/rnn/lstm-cell.png" | absolute_url }}){: .center-image }{:heights="500px"}    

* **4 network layers** as **yellow boxes**
    - each of them with their own weights
    - $$\sigma$$ is **sigmoid**
    - $$\tanh$$ is **hyperbolic tangent function**
        - similar to sigmoid that **squashes inputs**
        - output is $$[-1, 1]$$
* **Red circles** are **point-wise** and **element-wise** operations        
* **Cell state**, labeled as `C`
    - Goes through **LSTM** cell with **little interaction**
        - **allowing information** to flow easily through the cells.
    - Modified only **element-wise** operations which function as **gates**
    - **Hidden state** is calculated from the **cell state**
* **Forget Gate**
    - Network can learn to forget information that causes **incorrect predictions**. (output: 0)
    - Long range of information that is helpful. (output: 1)
    ![forget-gate.png]({{ "/assets/img/rnn/forget-gate.png" | absolute_url }}){: .center-image }{:heights="500px"}    
* **Update State**
    - this gate **updates** the **cell from the input** and **previous hidden state**
    ![update-state.png]({{ "/assets/img/rnn/update-state.png" | absolute_url }}){: .center-image }{:heights="500px"}
* **Cell State to Hidden Output** 
    - cell state is used to produce the **hidden state**, to next **hidden cell**.
    - Sigmoid gates let the network learn which information to keep and rid of.
    ![hidden-state.png]({{ "/assets/img/rnn/hidden-state.png" | absolute_url }}){: .center-image }{:heights="500px"}
    
## Character-wise RNN

- [From the Scratch](https://github.com/byam/dlnd/blob/master/4.RecurrentNeuralNetwork/intro-to-rnns/Anna_KaRNNa_Solution.ipynb)

## Resources

Here are a few great resources for you to learn more about recurrent neural networks. We'll also continue to cover RNNs over the coming weeks.

- [Andrej Karpathy's lecture](https://www.youtube.com/watch?v=iX5V1WpxxkY) on RNNs and LSTMs from CS231n
- [A great blog post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Christopher Olah on how LSTMs work.
- [Building an RNN](http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html) from the ground up, this is a little more advanced, but has an implementation in TensorFlow.
