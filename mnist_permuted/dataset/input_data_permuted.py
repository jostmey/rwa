def read_data_sets():

	basepath = '/'.join(__file__.split('/')[:-1])

	import input_data
	data = input_data.read_data_sets(basepath+'/bin', one_hot=True)

	import os
	import numpy as np
	if not os.path.isfile(basepath+'/bin/permutation.npy'):
		indices = np.random.permutation(28**2)
		os.makedirs(basepath+'/bin', exist_ok=True)
		np.save(basepath+'/bin/permutation.npy', indices)
	else:
		indices = np.load(basepath+'/bin/permutation.npy')

	data.train.images[:,:] = data.train.images[:,indices]
	data.validation.images[:,:] = data.validation.images[:,indices]
	data.test.images[:,:] = data.test.images[:,indices]

	return data

