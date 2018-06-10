#
# "counter.py" # Primary multiplication set
#
#  an example of keras (with tensorflow by Google)
#   by U.minor
#    free to use with no warranty
#
# usage:
# python counter.py 10000
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

argvs = sys.argv

dim = 20
samples = 500
predicts = 100

# generate sample data
i_train, o_train = [], []
for i in range(0, samples):
	x = []
	for j in range(0, dim):
		if uniform(0,2) < 1.0:
			a = 0
		else:
			a = 1
		x.append(a)

	le = len(x)
	for i, a in enumerate(x):
		if 0 < i and i < le - 1:
			if x[i - 1] == 1 and x[i + 1] == 1:
				x[i] = 1

	i_train.append(x)
	a_ = 0
	c = 0
	for a in x:
		if a_ == 0 and a == 1:
			c = c + 1
		a_ = a

	o_train.append(c)

i_train = np.array(i_train)

print(i_train)
print(o_train)


from keras.layers import Dense, Activation
model = keras.models.Sequential()

# neural network model parameters
hidden_units = 20
layer_depth = 1
act =  'sigmoid' 
bias = True

# first hidden layer
model.add(Dense(units = hidden_units, input_dim = dim, use_bias = bias))
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
	x = []
	for j in range(0, dim):
		if uniform(0,2) < 1.0:
			a = 0
		else:
			a = 1
		x.append(a)

	le = len(x)
	for i, a in enumerate(x):
		if 0 < i and i < le - 1:
			if x[i - 1] == 1 and x[i + 1] == 1:
				x[i] = 1

	i_test.append(x)
	a_ = 0
	c = 0
	for a in x:
		if a_ == 0 and a == 1:
			c = c + 1
		a_ = a

	o_true.append(c)
	if c > cmax:
		cmax = c

i_train = np.array(i_train)

i_test = np.array(i_test)
o_predict = model.predict(i_test)
o_true = np.array(o_true)

# Easy evaluation
for (it, op, ot) in zip(i_test, o_predict, o_true):
	print('{0} {1:>5.2f} {2} {3:>5.2f}'.format(it, op[0], ot, op[0] - ot))

amax = np.amax(o_predict)
if amax > cmax:
	cmax = int(amax)

plt.xticks(range(cmax + 2)) 
plt.yticks(range(cmax + 2)) 

plt.scatter(o_true, o_predict, c="blue", marker=".", s=200)
plt.show()

