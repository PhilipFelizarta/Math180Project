from model import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Remove logging messages in terminal
from time import time

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
from matplotlib import rc

import matplotlib.animation as animation
from matplotlib import style

from scipy.stats import norm #For our active learning algorithm

#We will use sklearn to create our Gaussian Process
import sklearn.gaussian_process as gp

#Load Data
model_input = np.load("sparse_model_input.npy")
expert_policy = np.load("sparse_expert_policy.npy")
state_value = np.load("sparse_state_value.npy")

#Shuffle the Data
np.random.seed(50) #Make this reproducible
shuffle_indices = np.random.permutation(len(model_input))
model_input = model_input[shuffle_indices]
expert_policy = expert_policy[shuffle_indices]
state_value = state_value[shuffle_indices]

#Reduce the size of these datasets.. we only need approximations for a quick search
model_input = model_input[:3626661]
expert_policy = expert_policy[:3626661]
state_value = state_value[:3626661]

#Parameters of our Search
iterations = 15 #How many parameters will we search before making a conclusion
exploration = 0.01

#Put nonsense values
hyper_params = []
cv_loss = []
indices = [] #Save indices of used hyper params for plotting

current_lr = 0.1
current_lambd = 1e-4
current_momentum = 0.8
current_value_weight = 1.0

#Generate hyperparameter grid space we will optimize over.
max_lr = 0.5
min_lr = 0.001

#Hyperparam for l2 regularization
max_lambd = 1e-1
min_lambd = 1e-5

#Coefficient on loss function for value  head
max_value_weight = 10.0
min_value_weight = 0.1

#Momentum of SGD optimizer
max_momentum = 0.9
min_momentum = 0.5

resolution = 15 #create a resxres grid of values to evaluate

#Calculate exponential coefficients for spacing
lr_coeff = np.power(min_lr / max_lr, 1/(resolution-1))
lambd_coeff = np.power(min_lambd / max_lambd, 1/(resolution-1))
value_weight_coeff = np.power(min_value_weight/ max_value_weight, 1/(resolution-1))

x_test = []
for x in range(resolution):
	lr = max_lr * np.power(lr_coeff, x)
	for y in range(resolution):
		lambd = max_lambd * np.power(lambd_coeff, y)
		for a in range(resolution):
			value_weight = max_value_weight * np.power(value_weight_coeff, a)
			for b in range(resolution):
				momentum = min_momentum + b * (max_momentum - min_momentum) / (resolution - 1)
				x_test.append([lr, lambd, value_weight, momentum])

x_test = np.reshape(x_test, [int(resolution*resolution*resolution*resolution), 4])

print("Testing: (", current_lr, ", ", current_lambd, ", ", current_value_weight, ", ", current_momentum, ")")

#Training Loop
plt.style.use('ggplot')
plt.ion()

fig, axs = plt.subplots(2, figsize=(8,8))
for it in range(iterations):

	#Conduct training to get an estimate of the generalization error of the model.
	start = time()
	_ , history = train_model(model_input, [expert_policy, state_value], 
	current_lr, current_lambd, current_value_weight, current_momentum, save=False, plot=False, load=False, summary=False, epochs=25, verbose=True) #Train a model using hypothesized best hyper params

	hyper_params.append([current_lr, current_lambd, current_value_weight, current_momentum]) #Append hyper params we used to train
	current_ce = history.history["val_policy_loss"][-1]
	current_mse = history.history["val_value_loss"][-1]
	current_loss = np.clip(current_ce + current_mse, 0, 10) #Sometimes the loss explodes... create a bound
	if np.isnan(current_loss):
		current_loss = 10.0
	cv_loss.append(current_loss) #Append final validation loss of model training
	print("--Actual Loss ", it, "--: " , current_loss)

	if it > 0:
		#Create a gaussian process to model what hyperparams are the best
		#kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
		kernel = gp.kernels.ConstantKernel(1.0) * gp.kernels.Matern(length_scale=1.0, nu=2.5)
		model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=False)

		x_train = np.reshape(hyper_params, [-1, 4])
		y_train = np.reshape(cv_loss, [-1, 1])
		model.fit(x_train, y_train)
		mu, sigma = model.predict(x_test, return_std=True) #Make a prediction of the loss over param space

		optimum = np.min(cv_loss)

		sigma = np.squeeze(sigma)
		mu = np.squeeze(mu)

		unbound_Z = (optimum - mu + exploration)/sigma
		Z = np.where(sigma > 0, unbound_Z, 0) #Calculate Z for EI formula
		unbound_EI = (optimum - mu + exploration)*norm.cdf(Z) + sigma*norm.pdf(Z)
		EI = np.where(sigma > 0, unbound_EI, 0)
		index = np.random.choice(np.flatnonzero(EI == np.max(EI))) #Randomly sample from the best.. if theres multiple
		current_lr = x_test[index][0]
		current_lambd = x_test[index][1]
		current_value_weight = x_test[index][2]
		current_momentum = x_test[index][3]

		indices.append(index)

		#Plot our estimations and our acqusition function
		axs[0].clear()
		axs[1].clear()
		axs[0].set_title("Bayesian Hyperparameter Optimization")
		axs[0].plot(mu, 'b--', lw=2, label="CV Loss (GP)")
		axs[0].fill_between(np.arange(int(resolution*resolution*resolution*resolution))
			, mu + sigma, mu - sigma, alpha=0.5, color="blue")
		axs[0].axhline(y=optimum, color="black", linestyle="--")
		axs[0].plot(indices, cv_loss, 'kx')
		axs[0].set_xlabel("Element of $ \mathcal{X} $")
		axs[0].legend(loc="upper right")
		axs[1].plot(EI, 'r', label="Expected Improvement")
		axs[1].axvline(x=index, color="black", linestyle="--")
		axs[1].set_xlabel("Element of $ \mathcal{X} $")
		axs[1].legend(loc="upper right")
		plt.savefig("figures/GPOptimization.png")
		plt.draw()
		plt.pause(0.0001)

		print("Testing: (", current_lr, ", ", current_lambd, ", ", current_value_weight, ", ", current_momentum, ")")
		print("Projected Loss: ", mu[index])
	else: #If we don't have enough samples for the gaussian process, randomly sample from the gridspace
		index = np.random.choice(int(resolution*resolution*resolution*resolution))
		current_lr = x_test[index][0]
		current_lambd = x_test[index][1]
		current_value_weight = x_test[index][2]
		current_momentum = x_test[index][3]
		indices.append(index)

		#Plot our estimations and our acqusition function
		axs[0].clear()
		axs[1].clear()
		axs[0].plot(indices, cv_loss, 'kx', label="Samples")
		axs[0].legend()
		axs[1].axvline(x=index, color="black", linestyle="--", label="Next Sample")
		axs[1].set_xlabel("Element of $ \mathcal{X} $")
		axs[1].legend()
		plt.draw()
		plt.pause(0.0001)

		print("Testing: (", current_lr, ", ", current_lambd, ", ", current_value_weight, ", ", current_momentum, ")")
		print("No Projected Loss (Random Sample)")

	print("Time (Minutes): ", str((time()-start)/60))


best_index = np.argmin(cv_loss)
best_params = hyper_params[best_index]
print("Best Parameters: ", best_params)
print("Sample Performance: ", np.min(cv_loss))

x_train = np.reshape(hyper_params, [-1, 4])
y_train = np.reshape(cv_loss, [-1, 1])

np.save("hyp_params", x_train)
np.save("cv_loss", y_train)
plt.savefig("Hyperparameter_Optimization.png")