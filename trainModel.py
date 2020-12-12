from model import *
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

import matplotlib.animation as animation
from matplotlib import style

from scipy.stats import norm #For our active learning algorithm


#Load Data
print("Loading...")
model_input = np.load("sparse_model_input.npy")
expert_policy = np.load("sparse_expert_policy.npy")
state_value = np.load("sparse_state_value.npy")

#Shuffle the Datas
print("Shuffling...")
np.random.seed(50) #Make this reproducible
shuffle_indices = np.random.permutation(len(model_input))
model_input = model_input[shuffle_indices]
expert_policy = expert_policy[shuffle_indices]
state_value = state_value[shuffle_indices]

print("Training...")
lr = 0.2497350799074219
lambd = 0.0002682695795279729
value_weight = 0.2682695795279727
momentum = 0.5285714285714286


_ , history = train_model(model_input, [expert_policy, state_value], lr, lambd, value_weight, momentum, 
save=True, plot=True, load=False, summary=True, epochs=25, verbose=True) #Train a model using hypothesized best hyper params

