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

# Configure dataset
#
S = 10
T = 1000
N_symbols = 8

# Data dimensions
#
num_train = 100000
num_test = 10000
max_length = 2*S+T
num_features = N_symbols+1
num_classes = N_symbols

##########################################################################################
# Generate/Load dataset
##########################################################################################

# Make directory
#
_path = '/'.join(__file__.split('/')[:-1])
os.makedirs(_path+'/bin', exist_ok=True)

# Training data
#
if not os.path.isfile(_path+'/bin/pattern_train.npy') or not os.path.isfile(_path+'/bin/recall_train.npy'):
	pattern_train = np.random.randint(num_features-2, size=(num_train, S))
	recall_train = 2*S+np.random.randint(1, T, size=num_train)
	np.save(_path+'/bin/pattern_train.npy', pattern_train)
	np.save(_path+'/bin/recall_train.npy', recall_train)
else:
	pattern_train = np.load(_path+'/bin/pattern_train.npy')
	recall_train = np.load(_path+'/bin/recall_train.npy')
xs_train = np.zeros((num_train, max_length, num_features))
ls_train = max_length*np.ones((num_train))
ys_train = np.zeros((num_train, max_length, num_classes))
xs_train[:,S:,num_features-2] = 1.0
ys_train[:,:,num_features-2] = 1.0
for i in range(num_train):
	for j in range(S):
		k = pattern_train[i,j]
		xs_train[i,j,k] = 1.0
		l = recall_train[i]
		xs_train[i,l-S-1,num_features-2] = 0.0
		xs_train[i,l-S-1,num_features-1] = 1.0
		ys_train[i,(l-S):l,:num_classes] = xs_train[i,:S,:num_classes]

# Test data
#
if not os.path.isfile(_path+'/bin/pattern_test.npy') or not os.path.isfile(_path+'/bin/recall_test.npy'):
	pattern_test = np.random.randint(num_features-2, size=(num_test, S))
	recall_test = 2*S+np.random.randint(1, T, size=num_test)
	np.save(_path+'/bin/pattern_test.npy', pattern_test)
	np.save(_path+'/bin/recall_test.npy', recall_test)
else:
	pattern_test = np.load(_path+'/bin/pattern_test.npy')
	recall_test = np.load(_path+'/bin/recall_test.npy')
xs_test = np.zeros((num_test, max_length, num_features))
ls_test = max_length*np.ones((num_test))
ys_test = np.zeros((num_test, max_length, num_classes))
xs_test[:,S:,num_features-2] = 1.0
ys_test[:,:,num_classes-1] = 1.0
for i in range(num_test):
	for j in range(S):
		k = pattern_test[i,j]
		xs_test[i,j,k] = 1.0
		l = recall_test[i]
		xs_test[i,l-S-1,num_features-2] = 0.0
		xs_test[i,l-S-1,num_features-1] = 1.0
		ys_test[i,(l-S):l,:num_classes] = xs_test[i,:S,:num_classes]

