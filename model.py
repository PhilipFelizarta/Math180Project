import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Remove logging messages in terminal

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Dropout, AlphaDropout, Lambda, Attention
from tensorflow.keras.layers import GlobalAveragePooling2D, Multiply, Permute, Reshape, Conv1D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt

def save_model(model, modelFile, weightFile): #Save model to json and weights to HDF5
	from tensorflow.keras.models import model_from_json


	model_json = model.to_json()
	with open(modelFile, "w") as json_file:
		json_file.write(model_json)
	model.save_weights(weightFile)
	print("Model Saved!")


def load_model(modelFile, weightFile):
	from tensorflow.keras.models import model_from_json

	json_file = open(modelFile, "r")
	load = json_file.read()
	json_file.close()

	model = model_from_json(load)
	model.load_weights(weightFile)
	print("Model Loaded")
	return model


def res_block(X, lr, lambd, filters=64):
	#Convolutional Layers
	channel = Conv2D(filters=int(filters), kernel_size=3, strides=1, 
	kernel_initializer=glorot_uniform(), kernel_regularizer=l2(lambd), activation="relu", padding="same")(X)
	channel = BatchNormalization(axis=-1)(channel)

	channel = Conv2D(filters=int(filters), kernel_size=3, strides=1, 
	kernel_initializer=glorot_uniform(), kernel_regularizer=l2(lambd), activation="linear", padding="same")(channel)
	channel = Squeeze_Excitation(channel, lambd, filters=filters)

	X = Add()([X, channel])
	X = Activation("relu")(X)
	X = BatchNormalization(axis=-1)(X)
	return X

def Squeeze_Excitation(X, lambd, filters=64, ratio=4):
	SE = GlobalAveragePooling2D()(X)
	SE = Dense(filters // ratio, activation="relu", kernel_regularizer=l2(lambd))(SE)
	SE = Dense(filters, activation="relu", kernel_regularizer=l2(lambd))(SE)
	SE = Reshape((1, 1, filters))(SE)
	SE = Multiply()([X, SE])
	return SE

def loss_metric(y_true, y_pred):
	#We have two outputs... get the tuple
	policy_true, value_true = y_true
	policy_pred, value_pred = y_pred

	CE = K.categorical_crossentropy(policy_true, policy_pred)
	MSE = K.mean_squared_error(value_true, value_pred)

	return CE + MSE



def create_model(lr, lambd, value_weight, momentum, filters=64, block_num=6, summary=False): #Convolutional Neural Network
	K.set_image_data_format('channels_last')
	image = Input(shape=(8,8,17)) #String State Representation

	#Convolutional Layers
	X = Conv2D(filters=filters, kernel_size=3, strides=1, 
	kernel_initializer=glorot_uniform(), kernel_regularizer=l2(lambd), activation="relu", input_shape=(10,26), padding="same")(image)

	for _ in range(block_num):
		X = res_block(X, lr, lambd, filters=filters)


	#Value head
	value = Conv2D(filters=32, kernel_size=1, strides=1,
		kernel_initializer=glorot_uniform(), kernel_regularizer=l2(lambd), activation="relu", padding="same")(X)
	value = BatchNormalization(axis=-1)(value)
	value = Flatten()(value)
	value = Dense(256, activation="relu", kernel_initializer=glorot_uniform(), 
		kernel_regularizer=l2(lambd))(value)
	value = Dense(1, activation="tanh", kernel_initializer=glorot_uniform(),
		kernel_regularizer=l2(lambd), name="value")(value)

	#Policy head
	policy = Conv2D(filters=32, kernel_size=1, strides=1,
		kernel_initializer=glorot_uniform(), kernel_regularizer=l2(lambd), activation="relu", padding="same")(X)
	policy = BatchNormalization(axis=-1)(policy)
	policy = Flatten()(policy)
	policy = Dense(1968, activation="softmax", kernel_initializer=glorot_uniform(),
		kernel_regularizer=l2(lambd), name="policy")(policy)

	#Create model now
	model = Model(inputs=image, outputs=[policy, value])


	#Learning Rate Schedule
	lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
		[15000.0, 30000.0, 45000.0, 60000.0],
		[lr, 0.1*lr, 0.01*lr, 0.001*lr, 0.0001*lr])

	#Compile model with our training hyperparameters
	optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=momentum, nesterov=True)
	model.compile(loss=["sparse_categorical_crossentropy", "mean_squared_error"], optimizer=optimizer,
	 metrics={"policy": "accuracy"}, loss_weights=[1.0, value_weight])
	
	if summary:
		model.summary() #Print a summary of the model
	return model

def train_model(x, y, lr, lambd, value_weight, momentum, save=False, plot=False, load=False, verbose=False, summary=False, epochs=10):
	model = create_model(lr, lambd, value_weight, momentum, summary=summary, filters=64)
	if load:
		start_param = load_model("Resnet.json", "Resnet.h5") #Import the trained model
		model.set_weights(start_param.get_weights())

	loss_history = model.fit(x, y, batch_size=1024, epochs=epochs, validation_split=0.1, verbose=verbose)

	if save:
		save_model(model, "long_ResNet64.json", "long_ResNet64.h5") #Save model

	if plot:
		plt.style.use('ggplot')
		
		plt.plot(loss_history.history['loss'], label='Training Loss')
		plt.plot(loss_history.history['val_loss'], label='Test Loss')
		plt.title('Loss for 6x64 Resnet')
		plt.ylabel('Loss')
		plt.xlabel('Epochs')
		plt.legend(loc="upper left")
		plt.savefig("figures/ModelLoss.png")
		plt.show()


		plt.plot(loss_history.history['policy_acc'], label='Accuracy (training data)')
		plt.plot(loss_history.history['val_policy_acc'], label='Accuracy (test data)')
		plt.title('Accuracy for 6x64 Resnet')
		plt.ylabel('Accuracy')
		plt.xlabel('Epochs')
		plt.legend(loc="upper left")
		plt.savefig("figures/ModelAccuracy.png")
		plt.show()


	return model, loss_history