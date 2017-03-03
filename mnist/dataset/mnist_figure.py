#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2017-02-16
# Purpose: Plot MNIST digits
# License: For legal information see LICENSE in the home directory.
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import matplotlib as mpl
mpl.use('Agg')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

##########################################################################################
# Settings
##########################################################################################

num_images = 25

##########################################################################################
# Load Data
##########################################################################################

import input_data
data = input_data.read_data_sets('bin', one_hot=True)

##########################################################################################
# Figure
##########################################################################################

fig = plt.figure()
for i in range(num_images):
	image = np.reshape(data.validation.images[i,:], [28, 28])
	ax = fig.add_subplot(1, num_images, i+1)
	ax.matshow(image, cmap=matplotlib.cm.binary)
	plt.xticks(np.array([]))
	plt.yticks(np.array([]))
plt.savefig('mnist_figure.png', dpi=300)

