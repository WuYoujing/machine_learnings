import numpy as np
import pandas as pd




def sigmoid(z):
    return 1/(1 + np.exp(-z))


def cost(theta,x,y,lam = 0):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    fir = np.multiply(-y,np.log(sigmoid(x @theta.T)))
    sec = np.multiply((1-y),np.log(1 - sigmoid(x@theta.T)))
    reg = (lam / (2 * len(x))) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum((fir - sec)/len(x)) + np.sum(reg)


def gradient(theta,x,y,lam = 0):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    para = theta.shape[1]
    grad = np.zeros(para)
    error = sigmoid(x*theta.T) - y
    
    for i in range(para):
        tem = np.multiply(error,x[:,i])
        if i == 0:
            grad[i] = np.sum(tem) / len(x)
        else:
            grad[i] = np.sum(tem)/len(x) + ((lam/len(x))*theta[:,i])
    
    return grad


def predict(theta,x):
    x = np.matrix(x)
    theta = np.matrix(theta)
    y = sigmoid(x*theta.T)
    y[y < 0.5] = 0
    y[y >= 0.5] = 1
    return y

def dec(theta2, x2, y2, rate):
    for i in range(50000):
        grad = gradient(theta2,x2,y2,rate)
        theta2 = theta2 - 0.01*grad
    return theta2
