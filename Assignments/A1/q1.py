# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 14:58:44 2019

@author: atezbas
"""

import numpy as np

# Data generation
def generate_gauss_classes(m,S,P,N):
  X = np.zeros((m.shape[1],N))
  y = np.zeros(N)
  for j in range(0,m.shape[1]):
    size = int(np.fix(P[j]*N))
    new_samples = np.random.multivariate_normal(m[j], S, size).T
    X[:, j*size: (j+1)*size] = new_samples
    y[j*size: (j+1)*size] = j+1
  return(X, y)

# Euclidian Classifier
def euclidian_classifier(m,X):
  c = m.shape[1]
  N = X.shape[1]
  de = np.zeros(c)
  z = np.zeros(N)
  for i in range(N):
    for j in range(c):
      de[j] = np.sqrt(np.dot((X[:,i]-m[j]),(X[:,i]-m[j])))
    z[i] = np.argwhere(de == np.min(de)) + 1
  return(z)

# Mahalonobis Classifier
def mahalonobis_classifier(m,S,X):
  c = m.shape[1]
  N = X.shape[1]
  dm = np.zeros(c)
  z = np.zeros(N)
  for i in range(N):
    for j in range(c):
      term1 = np.dot((X[:,i]-m[j]),np.linalg.inv(S))
      term2 = np.dot(term1, (X[:,i]-m[j]))
      dm[j] = np.sqrt(term2)
    z[i] = np.argwhere(dm == np.min(dm)) + 1
  return(z)

# Bayesian Classifier


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

# Generate a test set with new seed
[X_test, y_test] = generate_gauss_classes(m, S, P, N_samples)

z1 = euclidian_classifier(m,X_test)
z2 = mahalonobis_classifier(m,S,X_test)

print('Euclidian')
print(z1)
print('Mahalonobis')
print(z2)
print('y_test')
print(y_test)

# MLE estimates of means
print(np.mean(X[:,0:int(N_samples/3)], axis = 1))

# Covariance matrices
