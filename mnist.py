# Akhil Waghmare
# akhil@awlabs.technology

import keras
# make sure using Keras version 1.2.2 for coremltools support
print("Keras version: ", keras.__version__)

#----------------------------------------------------#

from keras.datasets import mnist
import matplotlib.pyplot as plt
# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

def sampleData():
	# plot 4 images from dataset in grayscale
	plt.subplot(221)
	plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
	plt.subplot(222)
	plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
	plt.subplot(223)
	plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
	plt.subplot(224)
	plt.imshow(X_train[32], cmap=plt.get_cmap('gray'))

	plt.show()
	
sampleData()

#----------------------------------------------------#

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

from keras import backend as K

# reshape data to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#----------------------------------------------------#

# model architecture:
# convolutional layer with 30 feature maps of size 5x5
# maxpooling layer w/ 2x2 patches
# convolutional layer with 15 feature maps of size 3x3
# maxpooling layer w/ 2x2 patches
# dropout w/ prob=0.2
# flatten
# fully connected layer w/ 128 neurons
# fully connected layer w/ 50 neurons
# output layer

# define CNN model
def model():
	model = Sequential()
	model.add(Conv2D(30, 5, 5, input_shape=(28, 28, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	
	#compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	return model
	
# train model
def trainModel(model):
	model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=128, validation_split=1/12., verbose=1)
	
# evaluate model
def testModel(model):
	scores = model.evaluate(X_test, y_test, verbose=0)
	print("Model accuracy on test data: %.2f%%" % (scores[1]*100))
	
# save model to file
def saveModel(model):
	model.save('mnistCNN.h5')

model = model()
trainModel(model)
#testModel(model)
saveModel(model)