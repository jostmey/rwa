#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2017-01-01
# Purpose: Load dataset or generate it if it does not exist yet.
# License: For legal information see LICENSE in the home directory.
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import os
import numpy as np

##########################################################################################
# Settings
##########################################################################################

# Data dimensions
#
num_train = 100000
num_test = 10000
max_length = 1000
num_features = 2

##########################################################################################
# Generate/Load dataset
##########################################################################################

# Make directory
#
_path = '/'.join(__file__.split('/')[:-1])
os.makedirs(_path+'/bin', exist_ok=True)

# Training data
#
if not os.path.isfile(_path+'/bin/xs_train.npy'):
	values = np.random.rand(num_train, max_length, 1)
	mask = np.zeros((num_train, max_length, 1))
	for i in range(num_train):
		j1, j2 = 0, 0
		while j1 == j2:
			j1 = np.random.randint(max_length)
			j2 = np.random.randint(max_length)
		mask[i,j1,0] = 1.0
		mask[i,j2,0] = 1.0
	xs_train = np.concatenate((values, mask), 2)
	np.save(_path+'/bin/xs_train.npy', xs_train)
else:
	xs_train = np.load(_path+'/bin/xs_train.npy')
ls_train = max_length*np.ones((num_train))
ys_train = np.sum(xs_train[:,:,0]*xs_train[:,:,1], 1)

# Test data
#
if not os.path.isfile(_path+'/bin/xs_test.npy'):
	values = np.random.rand(num_test, max_length, 1)
	mask = np.zeros((num_test, max_length, 1))
	for i in range(num_test):
		j1, j2 = 0, 0
		while j1 == j2:
			j1 = np.random.randint(max_length)
			j2 = np.random.randint(max_length)
		mask[i,j1,0] = 1.0
		mask[i,j2,0] = 1.0
	xs_test = np.concatenate((values, mask), 2)
	np.save(_path+'/bin/xs_test.npy', xs_test)
else:
	xs_test = np.load(_path+'/bin/xs_test.npy')
ls_test = max_length*np.ones((num_test))
ys_test = np.sum(xs_test[:,:,0]*xs_test[:,:,1], 1)

