---
layout: post
title:  "Autoencoders"
date:   2018-01-08 20:00:00 +0900
categories: dlnd
fbcomments: true
---

Outline:

- [Scratch](#scratch)
- [Autoencoders](#autoencoders)

## Scratch
- [A Simple Autoencoder](https://github.com/byam/dlnd/blob/master/3.ConvolutionalNeuralNetworks/autoencoder/Simple_Autoencoder_Solution.ipynb)
- [Convolutional Autoencoder](https://github.com/byam/dlnd/blob/master/3.ConvolutionalNeuralNetworks/autoencoder/Convolutional_Autoencoder_Solution.ipynb)

## Autoencoders

* Type of network architecture are used to **compress data**, as well as **image denoising**.
    - **compression** and **decompression** functions are learned from data itself.
* General idea
    - pass an **input data** through an **encoder**, to make **compressed representation**
    - then pass the **compressed representation** through a **decoder**, to get back reconstructed data
    - **encoder** and **decoder** are both built with **neural networks**.
    ![encoder-decoder.png]({{ "/assets/img/autoencoder/encoder-decoder.png" | absolute_url }}){: .center-image }{:heights="500px"}
    - The whole network is **trained by minimizing** the difference between input and output data.
    ![hidden-layer.png]({{ "/assets/img/autoencoder/hidden-layer.png" | absolute_url }}){: .center-image }{:heights="500px"}    
* Pros and cons
    - Pros
        - image **denoising**
        - dimensionality **reduction**
    - Cons
        - worse at **compression**. (jpeg, mp3 are better)
        - problems with **generalizing to datasets**.

![autoencoder-1.png]({{ "/assets/img/autoencoder/autoencoder-1.png" | absolute_url }}){: .center-image }{:heights="500px"}
