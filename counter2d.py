#
# "counter2d.py" # Primary multiplication set
#
#  an example of keras (with tensorflow by Google)
#   by U.minor
#    free to use with no warranty
#
# usage:
# python counter2d.py 10000
#
# last number (10000) means learning epochs, default=1000 if omitted

import tensorflow as tf
import keras
from keras.optimizers import SGD
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
import sys
import time

def flatten(nested_list):
	return [e for inner_list in nested_list for e in inner_list]

def set_dot(field, x, y):
#	field[x][y] = 1
	m = np.random.randint(2) + 1 # 2
	m2 = m ** 2
	for iy in range(y - m, y + m + 1):
		if 0 <= iy and iy <= field.shape[1] - 1 :
			for ix in range(x - m, x + m + 1):
				if 0 <= ix and ix <= field.shape[0] - 1 :
					if (ix - x) ** 2 + (iy- y) ** 2 < m2:
						field[ix][iy] = 1

def around(field, x, y):
#	return field[x][y] == 0

	m = 3
	m2 = m ** 2
	for iy in range(y - m, y + m + 1):
		if 0 <= iy and iy <= field.shape[1] - 1 :
			for ix in range(x - m, x + m + 1):
				if 0 <= ix and ix <= field.shape[0] - 1 :
					if (ix - x) ** 2 + (iy- y) ** 2 < m2:
						if field[ix][iy] == 1:
							return False

	return True

argvs = sys.argv

dim = 10
samples = 9000
predicts = 1000

# generate sample data
i_train, o_train = [], []

for i in range(0, samples):
	field = np.zeros((dim, dim))
	c = np.random.randint(dim ** 2)
	cc = 0
	for i in range(c):
		x, y = np.random.randint(dim), np.random.randint(dim)
		if around(field, x, y):
			set_dot(field, x, y) 
			cc += 1

	print(field)
	f = flatten(field)
	i_train.append(f)
	o_train.append(cc)

#sys.exit()

i_train = np.array(i_train)

#print(i_train)
#print(o_train)

from keras.layers import Dense, Activation
model = keras.models.Sequential()

# neural network model parameters
hidden_units = dim ** 2
layer_depth = 1
act =  'sigmoid' 
bias = True

# first hidden layer
model.add(Dense(units = hidden_units, input_dim = dim ** 2, use_bias = bias))
model.add(Activation(act))

#model.add(Dense(units = int(hidden_units / 2), use_bias = bias))
#model.add(Activation(act))

# output layer
model.add(Dense(units = 1, use_bias = bias))
model.add(Activation('linear'))

# Note: Activation is not 'softmax' for the regression model.

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss = 'mean_squared_error', optimizer = sgd)

# Note: loss is not 'sparse_categorical_crossentropy' for the regression model.
#        metrics = ['accuracy'] does not seem suitable.

# training
if len(argvs) > 1 and argvs[1] != '':
	ep = int(argvs[1]) # from command line
else:
	ep = 1000 # default

start_fit = time.time()

model.fit(i_train, o_train, epochs = ep, verbose = 1)
elapsed = time.time() - start_fit
print("elapsed = {:.1f} sec".format(elapsed))

# predict
i_test, o_true = [], []
cmax = 0

for i in range(0, predicts):
	field = np.zeros((dim, dim))
	c = np.random.randint(dim ** 2)
	cc = 0
	for j in range(c):
		x, y = np.random.randint(dim), np.random.randint(dim)
		#if field[x][y] == 0:
		if around(field, x, y):
			set_dot(field, x, y)
			cc += 1

	print(field)
	f = flatten(field)
	i_test.append(f)
	o_true.append(cc)
	if cc > cmax:
		cmax = cc
	print(cc)

i_test = np.array(i_test)
o_predict = model.predict(i_test)
o_true = np.array(o_true)

# Easy evaluation
for (it, op, ot) in zip(i_test, o_predict, o_true):
#	print('{0} {1:>5.2f} {2} {3:>5.2f}'.format(it, op[0], ot, op[0] - ot))
	print('{0:>5.2f} {1} {2:>5.2f}'.format(op[0], ot, op[0] - ot))

amax = np.amax(o_predict)
if amax > cmax:
	cmax = int(amax)

print(i_train.shape,i_test.shape)

plt.xticks(range(cmax + 2)) 
plt.yticks(range(cmax + 2)) 

plt.scatter(o_true, o_predict, c="blue", marker=".", s=200)
plt.show()

