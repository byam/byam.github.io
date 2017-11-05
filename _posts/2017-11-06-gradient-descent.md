---
layout: post
title:  "Gradient Descent"
date:   2017-11-06 14:50:00 +0900
categories: dlnd
fbcomments: true
---

Outline:
- [Reference](#reference)

## Learning weights

- We'll need to **learn the weights** from example data, then use those weights to make the predictions.
- We want the network to make predictions as close as possible to the real values.
- To measure this, we need a metric of how wrong the predictions are, **the error**. 
    - A common metric is the **sum of the squared errors (SSE)**:
    ![sse]({{ "/assets/img/gradient/sse.png" | absolute_url }}){: .center-image }{:width="400px"}

## Reference

- [Gradient tutorial of Khan Academy](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/gradient-and-directional-derivatives/v/gradient)
