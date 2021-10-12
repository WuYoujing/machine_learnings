import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import linalg


def normalEqnsaass(X,y):
    theta = np.linalg.inv(X.T*X)*X.T*y
    return theta


def gradintDescent(X,y,theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = theta.shape[1]
    cost = np.zeros(iters)
    for i in range(iters):
        error = (X * theta.T) - y
        for j in range(parameters):
            term = np.multiply(error,X[:,j])
            temp[0,j] = theta[0,j] - ((alpha/len(X))*sum(term))
        theta = temp
        cost[i] = computecost(X,y,theta)
    return theta, cost


def computecost(X,y,theta):
    inner = np.power(((X*theta.T)-y),2)
    return sum(inner)/(2*len(X))


