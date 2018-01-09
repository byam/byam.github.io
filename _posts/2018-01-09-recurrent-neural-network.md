---
layout: post
title:  "Recurrent Neural Network"
date:   2018-01-09 09:00:00 +0900
categories: dlnd
fbcomments: true
---

Outline:

- [Intro to RNN](#intro-to-rnn)


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
