import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ex1function

path = 'ex1data1.txt'

data = pd.read_csv(path, header=None,names=['Population','Profit'])
data.head()
data.describe()

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))


plt.scatter(data['Population'],data['Profit'])


plt.plot(data['Population'],data['Profit'],'o')
plt.show()

data.insert(0, 'ones',1)

col = data.shape[1]
X = data.iloc[:,0:col-1]
y = data.iloc[:,col-1:col]
X = np.matrix(X.values)

y = np.matrix(y.values)
theta = np.matrix([0,0])

print(ex1function.computecost(X,y,theta))

alpha = 0.01
iters = 1000

g,cost = ex1function.gradintDescent(X,y,theta,alpha,iters)

print(g)
print(cost)


x = np.linspace(data.Population.min(),data.Population.max(),100)
f = g[0,0] + (g[0,1] * x)
fig,ax = plt.subplot(figsize = (12,8))