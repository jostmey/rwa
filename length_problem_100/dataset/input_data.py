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
max_length = 100
num_features = 1

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
	xs_train = np.random.rand(num_train, max_length, num_features)
	np.save(_path+'/bin/xs_train.npy', xs_train)
else:
	xs_train = np.load(_path+'/bin/xs_train.npy')
if not os.path.isfile(_path+'/bin/ls_train.npy'):
	ls_train = np.random.randint(0, max_length, num_train)
	np.save(_path+'/bin/ls_train.npy', ls_train)
else:
	ls_train = np.load(_path+'/bin/ls_train.npy')
ys_train = np.round(ls_train.astype(float)/max_length)

# Test data
#
if not os.path.isfile(_path+'/bin/xs_test.npy'):
	xs_test = np.random.rand(num_test, max_length, num_features)
	np.save(_path+'/bin/xs_test.npy', xs_test)
else:
	xs_test = np.load(_path+'/bin/xs_test.npy')    
if not os.path.isfile(_path+'/bin/ls_test.npy'):
	ls_test = np.random.randint(0, max_length, num_test)
	np.save(_path+'/bin/ls_test.npy', ls_test)
else:
	ls_test = np.load(_path+'/bin/ls_test.npy')
ys_test = np.round(ls_test.astype(float)/max_length)

