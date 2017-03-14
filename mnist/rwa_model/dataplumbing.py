#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2017-01-01
# Purpose: Load dataset and create interfaces for piping the data to the model
# License: For legal information see LICENSE in the home directory.
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import numpy as np

##########################################################################################
# Class definitions
##########################################################################################

# Defines interface between the data and model
#
class Dataset:
	def __init__(self, xs, ls, ys):
		self.xs = xs	# Store the features
		self.ls = ls	# Store the length of each sequence
		self.ys = ys	# Store the labels
		self.num_samples = len(ys)
		self.num_features = len(xs[0,0,:])
		self.max_length = len(xs[0,:,0])
		self.num_classes = len(ys[0,:])
	def batch(self, batch_size):
		js = np.random.randint(0, self.num_samples, batch_size)
		return self.xs[js,:,:], self.ls[js], self.ys[js,:]

##########################################################################################
# Import dataset
##########################################################################################

# Load MNIST data
#
import sys
sys.path.append('../dataset')
import input_data
_data = input_data.read_data_sets('../dataset/bin', one_hot=True)

# Merge training and validation sets back together
#
_train_images = np.concatenate([_data.train.images, _data.validation.images])
_train_labels = np.concatenate([_data.train.labels, _data.validation.labels])

# Create split of data
#
train = Dataset(
	np.reshape(_train_images, [60000, 28**2, 1]),
	(28**2)*np.ones(60000),
	_train_labels
)
test = Dataset(
	np.reshape(_data.test.images, [10000, 28**2, 1]),
	(28**2)*np.ones(10000),
	_data.test.labels
)

