# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 14:58:44 2019

@author: atezbas
"""

import numpy as np

def generate_gauss_classes(m,S,P,N):
  X = np.zeros((m.shape[1],N))
  y = np.zeros(N)
  print(X.shape)
  for j in range(0,m.shape[1]):
    size = int(np.fix(P[j]*N))
    new_samples = np.random.multivariate_normal(m[j], S, size).T
    X[:, j*size: (j+1)*size] = new_samples
    y[j*size: (j+1)*size] = j+1

  print(y)
  return(X, y)

# Constant seed
np.random.seed(0)
# Number of samples
N_samples = 30
# Means
m = np.array([[0., 0., 0.],[1., 2., 2.],[3., 3., 4.]])
# Covariance Matrices
S = 0.8 * np.eye(3)
# Probabilities
P = np.array([1/3, 1/3, 1/3])
# Random data from multivariate_normal
[X, y] = generate_gauss_classes(m, S, P, N_samples)