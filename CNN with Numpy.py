#This is a test
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('train.csv')
data = np.array(data)
#this will give our rows and columns
m, n = data.shape
#randomly shuffling our data
np.random.shuffle(data)
#we will take the first 1000 samples and transpose them
data_dev = data[0:1000].T
Y_dev=data_dev[0]
X_dev=data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

def init_params():
    #This will generate random values between -0.5 and 0.5
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.randn(10 , 1) - 0.5
    W2 = np.random.randn(10, 10) - 0.5
    b2 = np.random.randn(10, 1) - 0.5
    return W1, W2, b1, b2

def ReLU(Z):
    return np.maximum(0, Z)

#forward propogation
def forward_prop(W1, W2, b1, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
