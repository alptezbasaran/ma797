# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 14:58:44 2019

@author: atezbas
"""

import numpy as np
import matplotlib.pyplot as plt

# Euclidian Classifier
def euclidean_classifier(m,X):
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

# Data generation
def generate_gauss_classes(m,S,P,N):
  X = np.zeros((m.shape[1],N))
  y = np.zeros(N)
  loc = 0
  for j in range(0,m.shape[1]):
    size = int(np.fix(P[j]*N))
    new_samples = np.random.multivariate_normal(m[j], S, size).T
    X[:, loc: loc + size] = new_samples
    y[ loc: loc + size] = j+1
    loc = loc + size
  return(X, y)

# Constant seed
np.random.seed(0)
# Number of samples
N_samples = 1000
# Means
m = np.array([[0., 0., 0.],[1., 2., 2.],[3., 3., 4.]])
# Covariance Matrices
S = 0.8 * np.eye(3)
# Probabilities
P = np.array([0.5, 0.25, 0.25])

# Random data from multivariate_normal
[X, y] = generate_gauss_classes(m, S, P, N_samples)

# Generate a test set with new seed
[X_test, y_test] = generate_gauss_classes(m, S, P, N_samples)


# ML Estimates
[m1_hat, S1_hat] = Gaussian_ML_estimate(X[:,np.where(y==1)[0]])
[m2_hat, S2_hat] = Gaussian_ML_estimate(X[:,np.where(y==2)[0]])
[m3_hat, S3_hat] = Gaussian_ML_estimate(X[:,np.where(y==3)[0]])
m_hat =np.vstack((m1_hat, m2_hat, m3_hat))
S_hat = 1/3*(S1_hat+*S2_hat+*S3_hat)

# Classifiers
z_euclidean = euclidean_classifier(m_hat,X_test)
z_mahalanobis = mahalanobis_classifier(m_hat,S_hat,X_test)
z_bayesian = bayes_classifier(m_hat,S_hat,P,X_test)


# KDE and Bayesian Classifier
def bayes_kde(X,X_test,y_test,bandwidth):
  from sklearn.neighbors import KernelDensity
  size = int(N_samples/3)
  pdf = np.empty((N_samples,3))

  for i in range(3):
    kde = KernelDensity(kernel='gaussian', bandwidth = bandwidth).fit(X[:, i*size: (i+1)*size].T)
    log_dens = kde.score_samples(X_test[:,      0:   size].T)
    class1 = np.exp(log_dens)
    log_dens = kde.score_samples(X_test[:,   size: 2*size].T)
    class2 = np.exp(log_dens)
    log_dens = kde.score_samples(X_test[:, 2*size: 3*size].T)
    class3 = np.exp(log_dens)
    pdf[i*size: (i+1)*size, :] = np.stack((class1, class2, class3), axis = 1)

  normalized_pdf = pdf/pdf.sum(axis=1)[:, np.newaxis]

  z_bayesian_kde = np.empty(N_samples)
  for i in range(N_samples):
    z_bayesian_kde[i] = bayes_classifier(m_hat,S_hat,normalized_pdf[i,:],X_test[:,i].reshape(3,1))

  return(z_bayesian_kde)

kde_error = []
bandwidth_range = np.linspace(0.05,5,100)
for i in range(len(bandwidth_range)):
  z_bayesian_kde = bayes_kde(X,X_test,y_test,bandwidth_range[i])
  err_kde_bayes = (1-len(np.where(y_test==z_bayesian_kde)[0])/len(y_test))
  kde_error.append(err_kde_bayes)

min_err_h = bandwidth_range[np.argwhere(kde_error == np.min(kde_error))]

plt.figure()
plt.plot(bandwidth_range,kde_error)
plt.ylabel('Error in Bayes Classifier')
plt.xlabel('h')
plt.title('Minimum Error is when h = {}'.format(min_err_h[0][0]))
plt.grid()

# h = 0.35
z_bayesian_kde = bayes_kde(X,X_test,y_test,min_err_h)

# kNN Classifier
knn_error = []
from sklearn.neighbors import KNeighborsClassifier
highest_k = 35
for i in range(1,highest_k+1):
  knn_classifier = KNeighborsClassifier(n_neighbors = i).fit(X.T,y.T)
  z_knn = knn_classifier.predict(X_test.T)
  knn_error.append(1-len(np.where(y_test==z_knn)[0])/len(y_test))

min_err_k = np.argwhere(knn_error == np.min(knn_error)) + 1

plt.figure()
plt.plot(list(range(1,highest_k+1)), knn_error)
plt.xlabel('Number of Neighbors')
plt.ylabel('Error in KNN Classifier')
plt.title('Minimum Error is when k = {}'.format(min_err_k[0][0]))
plt.grid()

# 8 neighbors
knn_classifier = KNeighborsClassifier(n_neighbors = min_err_k[0][0]).fit(X.T,y.T)
z_knn = knn_classifier.predict(X_test.T)

# Error estimates
print('Euclidean')
err_euclidean = (1-len(np.where(y_test==z_euclidean)[0])/len(y_test))
print(err_euclidean)
print('Mahalonobis')
err_mahalanobis = (1-len(np.where(y_test==z_mahalanobis)[0])/len(y_test))
print(err_mahalanobis)
print('Bayesian')
err_bayesian = (1-len(np.where(y_test==z_bayesian)[0])/len(y_test))
print(err_bayesian)
print('Bayesian - KDE')
err_kde_bayesian = (1-len(np.where(y_test==z_bayesian_kde)[0])/len(y_test))
print(err_bayesian)
print('KNN')
err_knn = (1-len(np.where(y_test==z_knn)[0])/len(y_test))
print(err_knn)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y.T, z_knn.T)
cm_bayes_kde = confusion_matrix(y.T, z_bayesian_kde.T)

