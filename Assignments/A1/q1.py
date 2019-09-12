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

# Mahalanobis Classifier
def mahalanobis_classifier(m,S,X):
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
def comp_gauss_dens_val(m,S,x):
#  print(m)
#  print(x)
#  print( np.exp(-0.5*np.dot(x-m ,np.dot(np.linalg.inv(S), x-m))))
  z = (1/( (2*np.pi)**(1/2)*np.linalg.det(S)**0.5)) * np.exp(-0.5*np.dot((x-m).T ,np.dot(np.linalg.inv(S), x-m)))
  return(z)

def bayes_classifier(m,S,P,X):
  c = m.shape[1]
  N = X.shape[1]
  z = np.zeros(N)
  t = np.zeros(c)
  for i in range(N):
    for j in range(c):
      t[j] = P[j] * comp_gauss_dens_val(m[j],S,X[:,i])
    z[i] = np.argwhere(t == np.max(t)) + 1
  return(z)

# Gaussian_ML_estimate
def Gaussian_ML_estimate(X):
  N = X.shape[1]
  l = X.shape[0]
  m_hat = 1/N * np.sum(X, axis = 1)
  S_hat = np.zeros((l,l))
  for k in range(N):
    diff = (X[:,k]-m_hat).reshape(3,1)
    S_hat = S_hat + np.dot(diff,diff.T)
  S_hat = 1/N * S_hat
  return(m_hat, S_hat)

# Constant seed
np.random.seed(0)
# Number of samples
N_samples = 999
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

# ML Estimates
[m1_hat, S1_hat] = Gaussian_ML_estimate(X[:,np.where(y==1)[0]])
[m2_hat, S2_hat] = Gaussian_ML_estimate(X[:,np.where(y==2)[0]])
[m3_hat, S3_hat] = Gaussian_ML_estimate(X[:,np.where(y==3)[0]])

z_euclidean = euclidian_classifier(m,X_test)
z_mahalanobis = mahalanobis_classifier(m,S,X_test)
z_bayesian = bayes_classifier(m,S,P,X)

# Error estimates
print('Euclidian')
err_euclidean = (1-len(np.where(y_test==z_euclidean)[0])/len(y_test))
print(err_euclidean)
print('Mahalonobis')
err_mahalanobis = (1-len(np.where(y_test==z_mahalanobis)[0])/len(y_test))
print(err_mahalanobis)
print('Bayesian')
err_bayesian = (1-len(np.where(y_test==z_bayesian)[0])/len(y_test))
print(err_bayesian)
