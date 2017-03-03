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
		self.num_samples = len(ls)
		self.num_features = len(xs[0,0,:])
		self.max_length = len(xs[0,:,0])
		self.num_classes = len(ys[0,0,:])
	def batch(self, batch_size):
		js = np.random.randint(0, self.num_samples, batch_size)
		return self.xs[js,:,:], self.ls[js], self.ys[js,:,:]

##########################################################################################
# Import dataset
##########################################################################################

# Load data
#
import sys
sys.path.append('../dataset')
import input_data

# Create split of data
#
train = Dataset(input_data.xs_train, input_data.ls_train, input_data.ys_train)
test = Dataset(input_data.xs_test, input_data.ls_test, input_data.ys_test)

