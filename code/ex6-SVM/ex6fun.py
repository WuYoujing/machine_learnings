import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat


def gaussin_kernal(x1,x2,sigma):
    return np.exp((-(x1-x2)@(x1-x2).T)/(2 * (sigma ** 2 )))

    