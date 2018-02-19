import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

my_data = pd.read_csv('data.txt',names=["x0","x1","y"]) #read the data


X = my_data.iloc[:,0:2]
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)
print(X)
y = my_data.iloc[:,2:3].values
theta = np.zeros([1,3])

alpha = 0.01
iters = 100000



def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))


def gradientDescent(X, y, theta, iters, alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha / len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost[i] = computeCost(X, y, theta)

    return theta, cost


# running the gd and cost function
g, cost = gradientDescent(X, y, theta, iters, alpha)
print(g)

finalCost = computeCost(X, y, g)
print(finalCost)

print(np.dot([1,644,444],g.T))

fig, ax = plt.subplots()
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()
