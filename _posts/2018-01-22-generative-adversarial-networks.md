---
layout: post
title:  "Generative Adversarial Networks"
date:   2018-01-22 00:00:00 +0900
categories: dlnd
fbcomments: true
---

Outline:

- [What you can do with GANs](#what-you-can-do-with-gans)
- [Other Generative Models](#other-generative-models)
- [Games and Equilibrium](#games-and-equilibrium)
- [GAN trained on the MNIST](#gan-trained-on-the-mnist)
- [More Learning Resources](#more-learning-resources)

## What you can do with GANs

GANs are used for **generating realistic data**.

Most of the applications of GANs so far have been for **images**.

* **stackGAN** model
    - taking a **textual description** of a bird, then generating a high resolution **photo of a bird** matching that description.
    - These photos have never been seen before, and are **totally imaginary**.
    ![stack-gan.png]({{ "/assets/img/gan/stack-gan.png" | absolute_url }}){: .center-image }{:width="400px"}
* A tool called **iGAN** developed in collaboration between Berkeley and Adobe
    - As a user **draws very crude sketches** using the mouse, iGAN searches for the nearest **possible realistic image**.
* image to image translation
    - where images in **one domain** can be turned into images in **another domain**.
    - image translation models can be trained in an **unsupervised way**.
        - Facebook, that can turn a photo of a face into a cartoon of a face.
        - Nvidia, For example to turn photos of day scenes into photos of night scenes.
    - At Berkeley a model called **CycleGAN** 
        - especially good at unsupervised image to image translation.
        - Here, it transforms this video of a horse to a video of a zebra.
        ![cycle-gan.png]({{ "/assets/img/gan/cycle-gan.png" | absolute_url }}){: .center-image }{:width="400px"}
*　GANs aren't limited to the visual domain.　
    - the outcome of high energy particle **physics experiments**.
    - Instead of using explicit Monte-Carlo simulation of the real physics of every step, the GAN learns by example what outcome is likely to occur in each situation.

## Other Generative Models

* **Generative adversarial networks** are a kind of **generative model**.
    - GANs are a kind of generative model that lets us generate a whole **image in parallel**.
    - GANs use a differentiable function, represented by a neural network as a **generator network**.
    - The **training process** for a generative model is very different from the training process for a supervised learning model.
        - GANs use an approximation where a second network, called the **discriminator**, learns to guide the generator.

![gan.png]({{ "/assets/img/gan/gan.png" | absolute_url }}){: .center-image }{:width="600px"}

* **Generator network**
    - takes **random noise** as input,
    - then runs that noise through a differentiable function to transform the noise and reshape it to have **recognizable structure**.
    - The output of the generator network is a **realistic image**.
    - The **goal** is for these images to be fair samples from the **distribution over real data**.

* **Discriminator**
    - The discriminator is just a regular neural net classifier,
    - During the training process
        - the discriminator is shown real images from the training data half the time
        - fake images from the generator the other half of the time.
        - The discriminator is trained to output the probability that the input is real.
        - Over time, the generator is forced to produce more realistic outputs in order to fool the discriminator.
 
## Games and Equilibrium

* To understand GANs, we need to think about 
    - **how payoffs** and **equilibria work** in the context of machine learning.
* If we can identify an equilibrium in the GAN game, 
    - we can use that equilibrium as a defining characteristic to understand that game.
    
* Cost
    - the cost for the discriminator is just the negative of the cost for the generator.
    - The generator wants to minimize the value function and the discriminator wants to maximize the value function.

* Saddle Point
    - That happens when we are at a maximum for the discriminator and a minimum for the generator.
    - the local maximum for the discriminator occurs
        - when the discriminator accurately estimates the probability that the input is real rather than fake.
        - This probability is given by the ratio between the data density at the input and the sum of both the data density and the model density induced by the generator at the input.
        - We can think of this ratio as measuring how much probability mass in an area comes from the data rather than from the generator.

* Equilibrium
    - where the generator density is equal to the true data density
    - and the discriminator outputs one half everywhere.
    - Unfortunately, even though the equilibrium exists, we may not be able to find it.  
    - We usually train GANs by running two optimization algorithms simultaneously
        - each minimizing one player's cost with respect to that player's parameters.
    - A common failure case for GANs is that when the data contains multiple clusters, 
        - the generator will learn to generate one cluster,
        - then the discriminator will learn to reject that cluster as being fake,
        - then the generator will learn to generate a different cluster, and so on.
        - We would prefer to have an algorithm that reliably finds the equilibrium where the generator samples from all the clusters simultaneously.
        
## GAN trained on the MNIST

* [Generative Adversarial Network](https://github.com/byam/dlnd/blob/master/5.GenerativeAdversarialNetworks/gan_mnist/Intro_to_GANs_Solution.ipynb)

## More Learning Resources:
   
- [introduction-generative-adversarial-networks-code-tensorflow](http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/)
- [generative-models](https://blog.openai.com/generative-models/)
- [karpathy/gan/](http://cs.stanford.edu/people/karpathy/gan/)
- [Generative-Adversarial-Networks](https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Generative-Adversarial-Networks)
- [gan-tensorflow](http://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/)
- [a-very-gentle-introduction-to-generative-adversarial-networks](https://www.slideshare.net/ThomasDaSilvaPaula/a-very-gentle-introduction-to-generative-adversarial-networks-aka-gans-71614428)
- [generative-adversarial-nets-in](http://blog.evjang.com/2016/06/generative-adversarial-nets-in.html)
- [generative-adversarial-networks-explained-with-a-classic-spongebob-squarepants-episode](https://medium.com/@awjuliani/generative-adversarial-networks-explained-with-a-classic-spongebob-squarepants-episode-54deab2fce39)
