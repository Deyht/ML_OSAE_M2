import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps

import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from sklearn import metrics
import tensorflow as tf


image_size = 128
nb_per_class = 11500
nb_test = 1000
augm_fact = 1
train_im = np.fromfile("train_im.dat", dtype="uint8")
train_im = np.reshape(train_im, ((nb_per_class*2*augm_fact, 3, image_size,image_size)))

train_data = np.zeros((nb_per_class*2*augm_fact,image_size,image_size,3), dtype="float32")
train_targets = np.zeros((nb_per_class*2*augm_fact,2), dtype="float32")

for i in range(0, nb_per_class*2*augm_fact):
	train_data[i] = train_im[i].T/255.0

train_targets[0:nb_per_class*augm_fact,0] = 1.0
train_targets[nb_per_class*augm_fact:,1]  = 1.0


test_im = np.fromfile("test_im.dat", dtype="uint8")
test_im = np.reshape(test_im, ((nb_test*2,3,image_size,image_size)))

test_data = np.zeros((nb_test*2,image_size,image_size,3), dtype="float32")
test_targets = np.zeros((nb_test*2,2), dtype="float32")

for i in range(0, nb_test*2):
	test_data[i] = test_im[i].T/255.0

test_targets[0:nb_test,0] = 1.0
test_targets[nb_test:,1]  = 1.0


######################### ##########################
#         Loading the neural network model
######################### ##########################

model = keras.Sequential()


model.add(layers.ZeroPadding2D(padding=(2, 2)))
model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(image_size,image_size,3)))
model.add(layers.MaxPooling2D())

model.add(layers.ZeroPadding2D(padding=(1, 1)))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D())

model.add(layers.ZeroPadding2D(padding=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D())

model.add(layers.Flatten())

model.add(layers.Dense(units=128, activation='relu'))
model.add(layers.Dense(units=2, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


######################### ##########################
#                 Network training
######################### ##########################

model.fit(train_data, train_targets, batch_size=16, epochs=1, shuffle=True,  validation_split=0.0, validation_data=(test_data, test_targets))

print(model.summary())

######################### ##########################
#            Evaluate the network prediction
######################### ##########################

model.evaluate(test_data, test_targets)

pred = model.predict(test_data)


matrix = metrics.confusion_matrix(test_targets.argmax(axis=1), pred.argmax(axis=1))

print (matrix)
















