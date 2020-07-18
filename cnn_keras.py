import numpy as np
import pickle
import cv2, os
from glob import glob
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import load_model
'''
K.common.set_image_dim_ordering('tf')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
'''
def get_image_size():
	img = cv2.imread('gestures/1/100.jpg', 0)
	return img.shape

def get_num_of_classes():
	return 11
	#return len(glob('gestures/*'))

image_x, image_y = get_image_size()


def plot_histories(histories):
	trainLoss = []
	valLoss =[]
	trainAcc = []
	valAcc = []
    #plt.clf()
	for history in histories:
		for TLoss in history.history['loss']:
			trainLoss.append(TLoss)
		for VLoss in history.history['val_loss']:
			valLoss.append(VLoss)
		for TAcc in history.history['accuracy']:
			trainAcc.append(TAcc)
		for VAcc in history.history['val_accuracy']:
			valAcc.append(VAcc)
	epoch = range(1, len(trainLoss) + 1)
	
	plt.plot(epoch, trainLoss, 'b', label='Training loss')
	plt.plot(epoch,valLoss,'r',label='Validation loss')
	plt.title('LOSS')
	plt.legend()
	plt.show()
	plt.plot(epoch, trainAcc, 'b', label='Training accuracy')
	plt.plot(epoch,valAcc,'r',label='Validation accuracy')
	plt.title('ACCURACY')
	plt.legend()
	plt.show()


def cnn_model():
	num_of_classes = get_num_of_classes()
	model = Sequential()
	#filters,strides,input_shape(x,y,channel),activation
	model.add(Conv2D(16, (2,2), input_shape=(image_x, image_y, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
	model.add(Conv2D(32, (3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
	model.add(Conv2D(64, (5,5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
	model.add(Flatten())
	#output shape
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(num_of_classes, activation='softmax'))
	sgd = optimizers.SGD(lr=1e-2)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	#initialize callback
	filepath="model.{epoch:02d}-{val_loss:.2f}.h5"
	checkpoint1 = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint1]

	from keras.utils import plot_model
	plot_model(model, to_file='model.png', show_shapes=True)
	return model, callbacks_list

def train():
	
	with open("train_images", "rb") as f:
		train_images = np.array(pickle.load(f))
	with open("train_labels", "rb") as f:
		train_labels = np.array(pickle.load(f), dtype=np.int32)

	with open("val_images", "rb") as f:
		val_images = np.array(pickle.load(f))
	with open("val_labels", "rb") as f:
		val_labels = np.array(pickle.load(f), dtype=np.int32)

	train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
	val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))
	train_labels = np_utils.to_categorical(train_labels)
	val_labels = np_utils.to_categorical(val_labels)
	
	print(val_labels.shape)
	
	model, callbacks_list = cnn_model()
	model.summary()
	initial_epoch=0
	while True:
		print("enter epoch number : ")
		epochs=int(input())
		#hist=model.fit(train_images, train_labels, validation_data=(val_images, val_labels), initial_epoch=initial_epoch,epochs=epochs, batch_size=500, callbacks=callbacks_list)
		histories = []
		histories.append(model.fit(train_images, train_labels, validation_data=(val_images, val_labels),initial_epoch=initial_epoch ,epochs=epochs, batch_size=500, callbacks=callbacks_list))

		#evaluate model
		scores = model.evaluate(val_images, val_labels, verbose=0)
		print("CNN Error: %.2f%%" % (100-scores[1]*100))

		plot_histories(histories)

		initial_epoch = epochs
	
	
	#save model
	#model.save('model.h5')
train()
K.clear_session();