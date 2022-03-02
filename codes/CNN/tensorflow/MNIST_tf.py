
# Neural networks pedagogical materials
# The following code is free to use and modify to any extent 
# (with no responsibility of the original author)

# Reference to the author is a courtesy
# Author : David Cornu => david.cornu@utinam.cnrs.fr


from tensorflow import keras
from tensorflow.keras import layers
from sklearn import metrics
import numpy as np
import tensorflow as tf




######################### ##########################
#          Loading data and pre process
######################### ##########################

data = np.fromfile("/obs/dcornu/CIANNA/mnist_dat/mnist_input.dat", dtype="float32")
data = np.reshape(data, (80000,28*28))
target = np.fromfile("/obs/dcornu/CIANNA//mnist_dat/mnist_target.dat", dtype="float32")
target = np.reshape(target, (80000,10))


data_train = np.reshape(data[:60000,:], (60000, 28,28,1))
data_valid = np.reshape(data[60000:70000,:], (10000, 28,28,1))
data_test  = np.reshape(data[70000:80000,:], (10000, 28,28,1))

target_train = target[:60000,:]
target_valid = target[60000:70000,:]
target_test  = target[70000:80000,:]


######################### ##########################
#         Loading the neural network model
######################### ##########################

model = keras.Sequential()


model.add(layers.ZeroPadding2D(padding=(2, 2)))
model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D())

model.add(layers.ZeroPadding2D(padding=(2, 2)))
model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
model.add(layers.MaxPooling2D())

model.add(layers.Flatten())

model.add(layers.Dense(units=128, activation='relu'))
model.add(layers.Dense(units=64, activation='relu'))
model.add(layers.Dense(units=10, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



######################### ##########################
#                 Network training
######################### ##########################

model.fit(data_train, target_train, batch_size=16, epochs=10, shuffle=True,  validation_split=0.0, validation_data=(data_valid, target_valid))



######################### ##########################
#            Evaluate the network prediction
######################### ##########################

model.evaluate(data_test, target_test)

pred = model.predict(data_test)


matrix = metrics.confusion_matrix(target_test.argmax(axis=1), pred.argmax(axis=1))

print (matrix)
















