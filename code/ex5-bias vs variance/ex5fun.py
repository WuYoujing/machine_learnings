import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    d = sio.loadmat()
    return map(np.ravel,[d['X'],d['y'],d['Xval'],d['yval'],d['Xtest'],d['ytest']])


def cost(theta,x,y):
    m = x.shape[0]


    co = x @ theta -y #r(m*1)
    co = co.T @ co
    co = co /(2*m)
    return co


def gradient(theta,x,y):
    m = x.shape[0]

    co = (x @ theta -y).T @ x #r(m*1)
    return co/m


def regularized_gradient(theta,x,y):
    m  = x.shape
    regularize = theta.copy()
    regularize[0] = 0
    regularize /= m
    return gradient(theta,x,y) + regularize


def lenear_regression_np(x,y,l=1):
    theta = np.zeros(x.shape[1])
    res = opt.minimize(fun = regularized_cost,
                        x0 = theta,
                        args = (x,y),
                        method = 'TNC',
                        jac =regularized_gradient,
                        options ={'disp':True})
    return res


def regularized_cost(theta, x, y, l=1):
    m = x.shape[0]

    regularized_term = (l / (2 * m)) * np.power(theta[1:], 2).sum()

    return cost(theta, x, y) + regularized_term


def poly_features(x,power,as_ndarray = False):
    data = { 'f{i}'.format(i):np.power(x,i) for i in range(1,power + 1)}
    df = pd.DataFrame(data)
    return df.as_matrix() if as_ndarray else df


def normalize_feature(df):
    """Applies function along input axis(default 0) of DataFrame."""
    return (df - df.mean()) / df.std()


def prepare_poly_data(*args,power):
    def prepare(x):
        df = poly_features(x,power)
        ndrr = normalize_feature(df)[:,:].values
        return np.insert(ndrr,0,np.ones(ndrr.shape[0],axis = 1))

    return prepare(x) for x in args


def plot_learning_curve(x,y,xval,yval,l = 0):
    trainingcost,cv_cost = [],[]

    m = x.shape[0]
    for i in range(1,m+1):
        res = lenear_regression_np(x[:i,:],y[:i],l = 1)
 
        tc = cost(res.x, x[:i, :], y[:i])
        cv = cost(res.x, xval, yval)
        trainingcost.append(tc)

        cv_cost.append(cv)

    fig, ax = plt.subplots(figsize = (20,12))
    ax = plot(range(1,m+1),trainingcost,label = 'trainingcost')
    ax = plot(range(1,m+1),cv_cost,label = 'cv cost')
    ax.legend(loc = 1)









