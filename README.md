## Description

This repository holds the code to a new kind of RNN model for processing sequential data. The model computes a recurrent weighted average (RWA) over each sequence to reduce the variable time-series data to a set of outputs. A detailed description of the RWA model has been published online at [URL].

In each folder, the RWA model is evaluated on a different task. The performance of the RWA model is compared against a LSTM model. The RWA is found to train faster and generalize better on each task.

## Download

* Download: [zip](https://github.com/jostmey/rwa/zipball/master)
* Git: `git clone https://github.com/jostmey/rwa`

## Requirements

The code is written in Python3. The models are implemented in TensorFlow using version 0.12.0-rc0.
