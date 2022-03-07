



import numpy as np
import matplotlib.pyplot as plt
#from numba import jit
#uncoment numba lines to get high speed up with on the fly compile

######################### ########################## ##########################
#                             ON-LINE FUNCTIONS
######################### ########################## ##########################

def error_fct(output, target):
	#the sum over all output neurons is made outside of the error fct
	return 0.5*(output-target)**2
	

#@jit(nopython=True, cache=True, fastmath=False)
def forward(input_vect, in_dim, hidden_vect, hid_dim, output_vect, out_dim, weights1, weights2, beta):
	######################### ##########################
	#        One forward step with a binary neuron
	######################### ##########################
	
	
	for i in range(0,hid_dim-1):
		h = np.sum(weights1[:,i]*input_vect[:])
		hidden_vect[i] = 1.0/(1.0 + np.exp(-beta*h))
	hidden_vect[hid_dim-1] = -1.0
		
	for i in range(0,out_dim):
		h = np.sum(weights2[:,i]*hidden_vect[:])
		output_vect[i] = 1.0/(1.0 + np.exp(-beta*h))
	
	
	
#@jit(nopython=True, cache=True, fastmath=False)
def backprop(input_vect, in_dim, hidden_vect, hid_dim, output_vect, targ_vect, out_dim, weights1, weights2, learn_rate, beta):
	######################### ##########################
	#       One backward step with a binary neuron
	######################### ##########################
	
	
	delta_o = beta*(output_vect[:] - targ_vect[:])*output_vect[:]*(1.0 - output_vect[:])
	delta_h = np.zeros(hid_dim-1)
	
	for i in range(hid_dim-1):
		h = np.sum(weights2[i,:]*delta_o[:])
		delta_h[i] = beta*hidden_vect[i]*(1.0-hidden_vect[i])*h
	
	
	for i in range(0, in_dim):
		weights1[i,0:hid_dim-1] -= learn_rate*(delta_h[0:hid_dim-1]*input_vect[i])
	
	for i in range(0, hid_dim):
		weights2[i,:] -= learn_rate*(delta_o[:]*hidden_vect[i])
	


def confmat(input, target, weights1, weights2, beta):
	######################### ##########################
	# Forward on an epoch and display a confusion matrix
	######################### ##########################
	
	in_dim = np.shape(input)[1]
	hid_dim = np.shape(weights2)[0]
	out_dim = np.shape(target)[1]
	nb_data = np.shape(input)[0]
	confmatrix = np.zeros((out_dim, out_dim))
	recall = np.zeros(out_dim)
	precis = np.zeros(out_dim)
	output = np.zeros(out_dim)
	hidden = np.zeros(hid_dim)
	
	accu = 0.0
	error = 0.0
	for i in range(0, nb_data):
	
		forward(input[i,:], in_dim, hidden, hid_dim, output, out_dim, weights1, weights2, beta)
		
		error += np.sum(error_fct(output[:], target[i,:]))
		
		max_a = np.argmax(output)
		max_b = np.argmax(target[i,:])
		confmatrix[max_b, max_a] += 1
		if(max_a == max_b):
			accu += 1
	
	for i in range(0,out_dim):
		recall[i] = 0
		precis[i] = 0
		for j in range(0,out_dim):
			recall[i] += confmatrix[i,j]
			precis[i] += confmatrix[j,i]
		
		if(recall[i] > 0.0):
			recall[i] = confmatrix[i,i] / recall[i] * 100.0
		
		if(precis[i] > 0.0):
			precis[i] = confmatrix[i,i] /precis[i] * 100.0
	
	print ("*****************************************************************")
	print ("Confmat :                                           Recall")
	for i in range(0,out_dim):
		print("         ", end="")
		for j in range(0,out_dim):
			print ("%10d"%confmatrix[i,j], end="")
		print("            %6.2f"%recall[i])
	print ("\n Precision  ", end="")
	for i in range(0,out_dim):
		print ("%10.2f"%precis[i], end="")
	print ("  Accu %8.2f%%"%(accu/nb_data*100.0))
	print ("\n*****************************************************************")
	
	return error/nb_data





######################### ##########################
#            User defined parameters
######################### ##########################


nb_epoch = VAL
control_interv = 10 #Only a display parameter
hid_dim = VAL
learn_rate = VAL
beta = VAL

######################### ##########################
#          Loading data and pre process
######################### ##########################

train_data = np.loadtxt("wine_train.data")
test_data = np.loadtxt("wine_test.data")

nb_train = np.shape(train_data)[0]
nb_test = np.shape(test_data)[0]

in_dim = np.shape(train_data)[1]-1
out_dim = 3

input = np.append(train_data[:,1:], -1.0*np.ones((nb_train,1)), axis=1)
input_test = np.append(test_data[:,1:], -1.0*np.ones((nb_test,1)), axis=1)
output = np.zeros(out_dim)
hidden = np.zeros(hid_dim+1)

target = np.zeros((nb_train,out_dim))
target_test = np.zeros((nb_test, out_dim))

for i in range(0,nb_train):
	target[i,int(train_data[i,0])-1] = 1.0
for i in range(0, nb_test):
	target_test[i,int(test_data[i,0])-1] = 1.0



######################### ##########################
#          Initialize network weights
######################### ##########################

weights1 = np.random.rand(in_dim+1,hid_dim)*(0.02)-0.01
weights2 = np.random.rand(hid_dim+1,out_dim)*(0.02)-0.01


######################### ##########################
#                Main training loop
######################### ##########################
######################### ##########################

for t in range(0,nb_epoch):

	if (((t%control_interv) == 0)):
		print("\n#########################################################################")
		print("\nIteration :", t)
		avg_error_test = confmat(input_test, target_test, weights1, weights2, beta)
		print( "Average test set error: ", avg_error_test)
	

	error = 0.0
	######################### ##########################
	#             Training on all data once
	######################### ##########################
	for i in range(0,nb_train):
		#Forward phase
		forward(input[i,:], in_dim+1, hidden, hid_dim+1, output, out_dim, weights1, weights2, beta)
		
		#Back-propagation phase
		backprop(input[i,:], in_dim+1, hidden, hid_dim+1, output, target[i,:], out_dim, weights1, weights2, learn_rate, beta)

		error += np.sum(error_fct(output[:],target[i,:]))
	
	avg_error_train = error/nb_train
	if((t%control_interv) == 0):
		print("\nAverage training dataset error :", avg_error_train)


######################### ##########################
######################### ##########################


print("\n#########################################################################")
print("\nFINAL RESULT at iteration :", t+1)
confmat(input_test, target_test, weights1, weights2, beta)










