# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:27:39 2019

@author: atezbas
"""

import numpy as np
import matplotlib.pyplot as plt


def plotter():
  plt.figure()
  plt.subplot(111)
  plt.subplots_adjust(right = 0.8)
  plt.ion()
  plt.scatter(X[0:4,1],X[0:4,2], marker = '+', s = 120)
  plt.scatter(X[4: ,1],X[4: ,2], marker = '_', s = 120)
  plt.xlim([-0.1, 2.1])
  plt.ylim([-0.1, 2.1])
  plt.xlabel(r'$x_1$')
  plt.ylabel(r'$x_2$')
  plt.title('Perceptron Iterations')
  plt.grid(linewidth=0.1)
  ax = plt.gca()
  return(ax)


def perceptron_sgd(X, Y, eta = 1, epochs = 20):
  ax = plotter()
  w = np.array([-2, 4, 1])

  for t in range(epochs):
    ax.plot([-0.1, 2.1],[-(w[0]+w[1] * -0.1)/w[2], -(w[0]+w[1] * 2.1)/w[2]], lw = 0.5, label = str(t))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, title = 'Epoch')
    plt.pause(0.1)
    for i, x in enumerate(X):
      if (np.dot(X[i], w)*Y[i]) <= 0:
        w = w + eta*X[i]*Y[i]
  return(w)


X = np.array([
  [0.0, 1.5],
  [1.0, 1.0],
  [2.0, 2.0],
  [2.0, 0.0],
  [0.0, 0.0],
  [1.0, 0.0],
  [0.0, 1.0],
])

X = np.insert(X, 0, 1, axis = 1) # Insert 1s to 0th column

y = np.array([1,1,1,1,-1,-1,-1])

w = perceptron_sgd(X,y, eta = 1,epochs = 10)
print(w)
