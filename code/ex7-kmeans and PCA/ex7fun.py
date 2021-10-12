import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

#该函数找到每一个样本里聚类中心最近的点
def find_centroid(x,centroid):
    m = x.shape[0]
    k = centroid.shape[0]
    idx = np.zeros(m)
    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((x[i,:] - centroid[j,:]))
            if dist < min_dist:
                min_dist = dist
                idx[i] = j
    return idx

def compute_centroids(x,idx,k):
    m,n = x.shape
    centroids = np.zeros((k,n))

    for i in range(k):
        indics = np.where(idx == i)
        centroids[i,:] = np.sum(x[idx == i,:],axis = 1)/len(idx[idx == i]).ravel()
    
    return centroids


def run_k_means(x, initial_centroids,max_iters):
    m,n = x.sahpe
    k = initial_centroids.shape[0]
    idx = np.zeros()
    centroids = initial_centroids

    for i in range(max_iters):
        idx = find_centroid(x,centroids)
        centroids = compute_centroids(x,idx,k)

    return idx,centroids


def init_centroids(x,k):
    m,n = x.shape
    centroids = np.zeros((k,n))
    idx = np.random.randint(0,m,k)
    for i in range(k):
        centroids[i,:] = x[idx[i],:] #从m个点中随机选取k个点作为初始的聚类中心
    return centroids


def pca(x):
    x = (x - x.mean()) / x.std()

    x = np.martix(x)

    cov = (x.T * x) /x.shape[0]

    u,s,v = np.liang.svd(cov)

    return u,s,v


def project_data(x,u,k):
    u_reduced = u[:,:k]
    return np.dot(x,u_reduced)

def recover_data(z,u,k):
    u_reduced = u[:,:k]
    return np.dot(z,u_reduced.T)






