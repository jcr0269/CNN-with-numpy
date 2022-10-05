
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
#The
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

def init_params():
    #This will generate random values between -0.5 and 0.5
    #Technically to get this though, you would need to take the mean -
    # of each value to get the range between 0.5 and -0.5 as its gaussian
    #Currently we have created a 2D array
    #W1 ndarray:(10,784)
    W1 = np.random.randn(10, 784)-0.5
    #b1 = nparray:(10,1)
    b1 = np.random.randn(10, 1) - 0.5
    #W2 (ndarray:(10,10)
    W2 = np.random.randn(10, 10) - 0.5
    b2 = np.random.randn(10, 1) - 0.5
    return W1, W2, b1, b2

#This will take the Max of 0 and Z, it is going through each element in Z
#if it is greater than 0 it will return Z, if less than then it will return 0
def ReLU(Z):
    return np.maximum(0, Z)

#preserves the amount of columns and collapes all of the rows.
#To get the probability that we want.
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
#read .txt of forward_prop to understand
def forward_prop(W1, W2, b1, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(A1)

#creates new matrix of zeros called one_hot_Y
#Y.size is just then the size of Y, and Y.max + 1 assuming that the zeros
# 0 - 9, by adding 1 then we get 0-10.
#np.arange that creates an array 0 - m for traning samples, Y is the lables
# the Y is what column it accesses for each row go to the column specified by Y and set
#it to 1
def one_hot(Y):
    one_hot_Y = np.zeros(Y.size, Y.max() + 1)
    one_hot_Y[np.arange(Y.size), Y] = 1
    #this just transposes the one_hot_Y or flips it.
    #each column not each row which is why we flip
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0
#Read paper on back_prop to understand more.
def back_prop(Z1, A1, Z2, A2, W2, Y, X):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    dB2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    dB1 = 1 / m * np.sum(dZ2, 2)
    return dW1, dW2, dB1, dB2

def update_params(W1, b1, W2, b2, dB1, dW1, dW2, dB2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * dB1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * dB2
    return W1, b1, b2, W2


def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size
def gradiant_descent(X, Y, alpha, iterations):
    W1, b1, b2, W2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        #every 10th itiration we print iteration.
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2