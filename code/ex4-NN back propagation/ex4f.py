import pandas as pd
import  numpy as np
from scipy.io import  loadmat


def sigmoid(z):
    return 1 /(1+ np.exp(-z))


def forward_propagate(x,theta1,theta2):
    m = x.shape[0]
    a1 = np.insert(x, 0, values = np.ones(m), axis=1)
    a1 = np.matrix(a1)
    theta1 = np.matrix(theta1)
    z2 = a1 @ theta1.T
    a2 = np.insert(sigmoid(z2), 0, values = np.ones(m), axis=1)
    theta2 = np.matrix(theta2)
    z3 = a2 @ theta2.T
    h = sigmoid(z3)
    return a1,z2,a2,z3,h

def cost(params, input_size, hidden_size, num_labels, x, y, learning_rate = 0):
    m = x.shape[0]
    x = np.matrix(x)
    y = np.matrix(y)

    theta1 = np.matrix(np.reshape(params[:hidden_size*(input_size+1)],(hidden_size,(input_size+1))))
    theta2 = np.matrix(np.reshape(params[hidden_size*(input_size+1):],(num_labels,(hidden_size+1))))
    a1,z2,a2,z3,h = forward_propagate(x,theta1,theta2)

    j = 0
    for i in range(m):
        fir = -y[i,:] @ np.log(h[i,:]).T
        sec = (1-y[i,:]) @ np.log(1-h[i,:]).T
        j = j + fir-sec
    j = j / m
    j = j + (float(learning_rate/(2*m))*(np.sum(np.power(theta1[:,1:],2)+np.power(theta2[:,1:],2))))
    return j 


def sigmoid_gradient(z):
    return np.multiply(sigmoid(z) , (1-sigmoid(z)))

def back_propagate(params,input_size,hidden_size,num_labels,x,y,learning_rate):
    x = np.matrix(x)
    y = np.matrix(y)
    m = x.shape[0]

    theta1 = np.matrix(np.reshape(params[:hidden_size*(input_size+1)],(hidden_size,(input_size+1))))
    theta2 = np.matrix(np.reshape(params[hidden_size*(input_size+1):],(num_labels,(hidden_size+1))))
    a1,z2,a2,z3,h = forward_propagate(x,theta1,theta2)

    j = 0
    for i in range(m):
        fir = -y[i,:] @ np.log(h[i,:]).T
        sec = (1-y[i,:]) @ np.log(1-h[i,:]).T
        j = j + fir-sec
    j = j / m
    j = j + (float(learning_rate/(2*m))*(np.sum(np.power(theta1[:,1:],2)+np.power(theta2[:,1:],2))))
    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)
    for t in range(m):
        a1t = a1[t,:]
        z2t = z2[t,:]
        a2t = a2[t,:]
        ht = h[t,:]
        yt = y[t,:]

        d3t = ht - yt        
        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T*d3t).T,sigmoid_gradient(z2t))
        delta2 = delta2 + d3t.T * a2t
        delta1 = delta1 + d2t[:,1:].T * a1t
        delta1 = delta1 / m
        delta2 = delta2 / m
        delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * learning_rate) / m
        delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * learning_rate) / m
        grad = np.concatenate(np.ravel(delta1),np.ravel(delta2))
        return j,grad

