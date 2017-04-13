## Description

This repository holds the code to a new kind of RNN model for processing sequential data. The model computes a recurrent weighted average (RWA) over every previous processing step. With this approach, the model can form direct connections anywhere along a sequence. This stands in contrast to traditional RNN architectures that only use the previous processing step. A detailed description of the RWA model has been published in a manuscript at [https://arxiv.org/pdf/1703.01253.pdf](https://arxiv.org/pdf/1703.01253.pdf).

![alt text](artwork/figure.png "Comparison of RNN architectures")

Because the RWA can be computed as a running average, it does not need to be completely recomputed with each processing step. The numerator and denominator can be saved from the previous step. Consequently, the model scales like that of other RNN models such as the LSTM model.

In each folder, the RWA model is evaluated on a different task. The performance of the RWA model is compared against a LSTM model. The RWA is found to train considerably faster on most tasks by at least a factor of five. As the sequences become longer, the RWA model scales even better. See the manuscript listed above for the details about each result.

## Download

* Download: [zip](https://github.com/jostmey/rwa/zipball/master)
* Git: `git clone https://github.com/jostmey/rwa`

## Requirements

The code is written in Python3. The scripts have been upgraded to run using version 1.0 of TensorFlow.

## Alternative Implementations

 * [RWA model as TensorFlow RNNCell](https://github.com/jostmey/cas/blob/master/RWACell.py) (Works but still under development)
 * [RWA model in Keras](https://gist.github.com/shamatar/55b804cf62b8ee0fa23efdb3ea5a4701) (Not yet tested)
 * [RWA model in Go](https://github.com/unixpickle/rwa)

## Acknowledgements

Thanks [Alex Nichol](https://github.com/unixpickle) for correcting the equations for numerical stability.

