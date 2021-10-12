import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


def cost(params,y,r,num_features,rate = 0):
    y = np.matrix(y)
    r = np.matrix(r)

    num_movies = y.shape[0]
    num_users = y.shape[1]

    x = np.matrix(np.reshape(params[:num_movies*num_features],(num_movies,num_features)))
    theta = np.matrix(np.reshape(params[num_movies*num_features:],(num_users,num_features)))

    j = 0
    error = np.multiply(x*theta.T - y, r)
    squared_error = np.power(error,2)
    j = (1/2) * np.sum(squared_error)

    x_grad = error * theta
    theta_grad = error.T * x
    j = j + ((rate / 2) * np.sum(np.power(theta, 2)))
    j = j + ((rate / 2) * np.sum(np.power(x, 2)))

    grad = np.concatenate((np.ravel(x_grad),np.ravel(theta_grad)))

    return j,grad


def estimate_gaussin(x):
    mu = x.mean(axis = 0)
    sigma = x.var(axis = 0)
    return mu,sigma


def select_threshold(pval,yval):
    best_epsilon = 0
    best_f1 = 0
    f1 = 0

    step = (pval.max()-pval.min())/1000
    for epsion in np.arange(pval.min(),pval.max(),step):
        preds = pval < epsion
        tp = np.sum(np.logical_and(preds == 1,yval == 1)).astype(float)
        fp = np.sum(np.logical_and(preds == 1 ,yval == 0)).astype(float)
        fn = np.sum(np.logical_and(preds == 0,yval == 1)).astype(float)

        precision = tp / ( tp + fp)
        recall = tp / ( tp + fn)
        f1 = (2 * precision *recall) / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsion
    return best_epsilon, best_f1


