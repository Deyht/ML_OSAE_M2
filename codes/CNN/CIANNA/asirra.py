import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps

import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'/obs/dcornu/CIANNA/src/build/lib.linux-x86_64-3.7')
import CIANNA as cnn

def i_ar(int_list):
	return np.array(int_list, dtype="int")

def f_ar(float_list):
	return np.array(float_list, dtype="float32")
	
	

cnn.init_network(dims=np.array([128,128,1,3]), out_dim=2, bias=0.1, b_size=16,comp_meth="C_BLAS")

image_size = 128
nb_per_class = 11500
nb_test = 1000
augm_fact = 6
train_im = np.fromfile("train_im.dat", dtype="uint8")
train_im = np.reshape(train_im, ((nb_per_class*2*augm_fact, image_size*image_size*3)))

train_data = np.zeros((nb_per_class*2*augm_fact,image_size*image_size*3), dtype="float32")
train_targets = np.zeros((nb_per_class*2*augm_fact,2), dtype="float32")

train_data[:,:] = train_im[:,:]/255.0

train_targets[0:nb_per_class*augm_fact,0] = 1.0
train_targets[nb_per_class*augm_fact:,1]  = 1.0




test_im = np.fromfile("test_im.dat", dtype="uint8")
test_im = np.reshape(test_im, ((nb_test*2, image_size*image_size*3)))

test_data = np.zeros((nb_test*2,image_size*image_size*3), dtype="float32")
test_targets = np.zeros((nb_test*2,2), dtype="float32")

test_data[:,:] = test_im[:,:]/255.0

test_targets[0:nb_test,0] = 1.0
test_targets[nb_test:,1]  = 1.0


cnn.create_dataset("TRAIN", nb_per_class*augm_fact*2, train_data[:,:], train_targets[:,:], silent=0)
cnn.create_dataset("VALID", nb_test*2, test_data[:,:], test_targets[:,:], silent=0)
cnn.create_dataset("TEST",  nb_test*2, test_data[:,:], test_targets[:,:], silent=0)


cnn.conv_create(f_size=i_ar([3,3,1]), nb_filters=32, stride=i_ar([1,1,1]), padding=i_ar([1,1,0]), activation="RELU")
cnn.pool_create(p_size=i_ar([2,2,1]), p_type="MAX")
cnn.conv_create(f_size=i_ar([3,3,1]), nb_filters=64, stride=i_ar([1,1,1]), padding=i_ar([1,1,0]), activation="RELU")
cnn.pool_create(p_size=i_ar([2,2,1]), p_type="MAX")
cnn.conv_create(f_size=i_ar([3,3,1]), nb_filters=64, stride=i_ar([1,1,1]), padding=i_ar([1,1,0]), activation="RELU")
cnn.pool_create(p_size=i_ar([2,2,1]), p_type="MAX")
cnn.conv_create(f_size=i_ar([3,3,1]), nb_filters=128, stride=i_ar([1,1,1]), padding=i_ar([1,1,0]), activation="RELU")
cnn.pool_create(p_size=i_ar([2,2,1]), p_type="MAX")
cnn.dense_create(nb_neurons=512, activation="RELU", drop_rate=0.2)
cnn.dense_create(nb_neurons=2, activation="SOFTMAX")

for t in range(0,1):
	cnn.train_network(nb_epoch=20, learning_rate=0.0001, end_learning_rate=0.00005, control_interv=1, 
		momentum=0.8, decay=0.015, confmat=1, shuffle_gpu=0, save_each=20, shuffle_every=1)
	
	if(t == 0):
		cnn.perf_eval

exit()














