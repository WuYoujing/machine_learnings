import numpy as np
import pandas as pd
from scipy.optimize import minimize


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
    para = int(theta.ravel().shape[1])
    error = sigmoid(x*theta.T) - y
    # for i in range(para):
    #     tem = np.multiply(error,x[:,i])
    #     if i == 0:
    #         grad[i] = np.sum(tem) / len(x)
    #     else:
    #         grad[i] = np.sum(tem)/len(x) + ((lam/len(x))*theta[:,i])
    grad = (x.T *error/len(x)).T + (lam/len(x))*theta
    grad[0,0] = np.sum(np.multiply(error,x[:,0]))/len(x)
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


def one_vs_all(x,y,numlabels,rate):
    rows = x.shape[0]
    clu = x.shape[1]

    thetas = np.zeros((numlabels,clu+1))
    x = np.insert(x, 0, values=np.ones(rows), axis=1)
    x = np.matrix(x)
    for i in range(1,numlabels+1):
        # y是一个列向量，保存了整数从1到10
        # y_i1 = np.array(y)
        # y_i1[y_i1 != i] = 0
        # y_i1[y_i1 == i] = 1
        # 为什么两种生成y的方式出来的数据都是一模一样的，但是结果截然不同？？？
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))
        theta = np.zeros(clu+1)
        fmin = minimize(fun = cost,x0 = theta,args=(x,y_i,rate), method='TNC',jac = gradient)
        thetas[i-1,:] = fmin.x
    return thetas


def predict_all(x,all_theta):
    rows = x.shape[0]
    cli = x.shape[1]
    numlabels = all_theta[0]


    x = np.insert(x,0,values=np.ones(rows),axis = 1)
    all_theta = np.matrix(all_theta)
    x = np.matrix(x)
    h = sigmoid(x*all_theta.T)
    h_arg = h.argmax(axis = 1)+1
    return h_arg








        