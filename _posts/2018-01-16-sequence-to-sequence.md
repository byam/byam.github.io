---
layout: post
title:  "Sequence to Sequence"
date:   2018-01-16 00:00:00 +0900
categories: dlnd
fbcomments: true
---

Outline:

- [Introduction](#introduction)
- [Architectures](#architectures)


## Introduction

#### Different kinds of RNNs

That are suited for different types of tasks.

* Many to One
    - The sentiment analysis RNN
    - It reads a sequence of words, and then outputs just a single value.
* Many to Many
    - a chat bot or a translation service
    - need **sequential inputs** and **sequential outputs**
* Sequence to Sequence (two RNNS)
    - one that reads the input sequence, 
    - then hands over what it had learned to another RNN, 
    - which starts producing the output sequence.

#### Applications

* Seq2seg model
    - Can learn to generate **any sequence of vectors** after we feed it a sequence of input vectors.
        - letters, words or images or anything.
    - Example
        - English-to-French translator
            - input: English phrase
            - target: French phrase
        - Summarization bot
            - input: dataset of questions
            - target: answers

## Architectures

* High level, the inference process
    - inputs to the encoder.
        - encoder summarizes what it understood into a context variable or state.
    - And it hands it over to the decoder,
        - which then proceeds to generate the output sequence.
