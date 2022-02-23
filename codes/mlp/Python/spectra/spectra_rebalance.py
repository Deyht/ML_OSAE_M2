

# Neural networks pedagogical materials
# The following code is free to use and modify to any extent 
# (with no responsibility of the original author)

# Reference to the author is a courtesy
# Author : David Cornu => david.cornu@utinam.cnrs.fr



import numpy as np
import sys
sys.path.insert(0,'..')
from ext_module import *


nb_epochs = 2000
control_interv = 50
hid_dim = 8
learn_rate = 0.02
out_dim = 7
beta = 1.0

nb_train = 150 # rebalanced over 1115
nb_test = 200
class_balance = [10,30,30,20,20,30,10]

######################### ##########################
#          Loading data and pre process
######################### ##########################

raw_data = np.loadtxt("../../../data/stellar_spectra/train.dat")
raw_target = np.loadtxt("../../../data/stellar_spectra/target.dat")


nb_dat = np.shape(raw_data)[0]
in_dim = np.shape(raw_data)[1]

input = np.append(raw_data[:,:in_dim], -1.0*np.ones((nb_dat,1)), axis=1)
output = np.zeros((nb_train, out_dim))
delta_o = np.zeros((nb_train, out_dim))
hidden = np.zeros((nb_train, hid_dim+1))
delta_h = np.zeros((nb_train, hid_dim+1))
output_test = np.zeros((nb_test, out_dim))
hidden_test = np.zeros((nb_test, hid_dim+1))


targ = raw_target

id_classes = []
for i in range(0, out_dim):
	temp = np.where(np.argmax(targ[:,:], axis=1) == i)
	id_classes.append(temp)


input[:,:-1] -= np.mean(input[:,:-1], axis = 0)
input[:,:-1] /= np.max(np.abs(input[:,:-1]), axis = 0)

# split training and test dataset
input_test = input[915:,:]
targ_test = targ[915:,:]

input_train = np.empty((0,in_dim+1))
targ_train = np.empty((0,out_dim))


for i in range(0,out_dim):
	index_list = id_classes[i][0][:class_balance[i]]
	print (index_list)
	for j in range(0, len(index_list)):
		input_train = np.append(input_train, np.reshape(input[index_list[j],:],(1,in_dim+1)), axis=0)
		targ_train = np.append(targ_train, np.reshape(targ[index_list[j],:],(1,out_dim)), axis=0)

nb_dat = nb_train


input = input_train
targ = targ_train



index = np.arange(0,nb_dat)

######################### ##########################
#          Initialize network weights
######################### ##########################

weights1 = np.zeros((in_dim+1,hid_dim+1))

weights1[:,:-1] = np.random.rand(in_dim+1,hid_dim)*(0.02)-0.01
weights1[in_dim,hid_dim] = 1.0;
weights2 = np.random.rand(hid_dim+1,out_dim)*(0.02)-0.01


######################### ##########################
#                Main training loop
######################### ##########################
######################### ##########################

for t in range(0,nb_epochs):

	if ((((t+1)%control_interv) == 0) or (t == 0)):
		print("\n#########################################################################")
		print("\nIteration :", t+1)
		confmat_batch(input_test, hidden_test, output_test, targ_test, weights1, weights2, beta)

	np.random.shuffle(index)
	input = input[index,:]
	targ = targ[index,:]

	quad_error = 0.0
	######################### ##########################
	#             Training on all data once
	######################### ##########################

	#Forward phase
	forward_batch(input, hidden, output, weights1, weights2, beta)
	
	#Back-propagation phase
	backprop_batch(input, hidden, delta_h, output, delta_o, targ, weights1, weights2, learn_rate, beta)

	quad_error = 0.5*np.sum((output[:,:] - targ[:,:])**2)
	
	if((t%control_interv) == 0):
		print("\nAverage training dataset quadratic error :", quad_error/nb_dat)

######################### ##########################
######################### ##########################














