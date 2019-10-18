# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:52:21 2019

@author: atezbas
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt

# Data
X = np.array([
  [1., 3.],
  [2., 3.],
  [2., 4.],
  [3., 1.],
  [3., 2.],
  [4., 2.],
])

y = np.array([1,1,1,-1,-1,-1])


# A2 Q1
X = np.array([
  [0.0, 1.5],
  [1.0, 1.0],
  [2.0, 2.0],
  [2.0, 0.0],
  [0.0, 0.0],
  [1.0, 0.0],
  [0.0, 1.0],
])

y = np.array([1,1,1,1,-1,-1,-1])

n1 = len(X[ y ==  1])         # Number of points label ' 1'
n2 = len(X[ y == -1])         # Number of points label '-1'

# LDA Implementation
np.set_printoptions(2)

# Calculate Means of Each Feature [2 x 2] 2 classes, 2 features
means = np.array((np.mean(X[y == 1], axis = 0),np.mean(X[y == -1], axis = 0)))
means

m1 = np.mean(X[y == 1], axis = 0)
m2 = np.mean(X[y == -1], axis = 0)

m1
m2

# Calculate the Covariance Matrices [2, 2]
var1 = 1/(n1 - 1) * np.dot((X[ y ==  1] - m1).T,(X[ y ==  1] - m1))     # (X[ y ==  1] - m1).shape = [3x2] 3 points, 2 features
var2 = 1/(n2 - 1) * np.dot((X[ y == -1] - m2).T,(X[ y == -1] - m2))     # (X[ y == -1] - m1).shape = [3x2] 3 points, 2 features

# Shared Covariance Matrix
S = n1 * var1 + n2 * var2
S
# Estimate the weights
w_new = np.dot(np.linalg.inv(S), m1-m2)

w0_new = np.zeros(1)
for i in range(X.shape[0]):
  w0_new += y[i]-np.dot(w_new.T,X[i,:])

w0_new = w0_new/X.shape[0]

for i in range(X.shape[0]):
  predict = np.sign(w0_new + np.dot(w_new.T,X[i,:]))
  print('Ayak Guzel mi =', predict)

print('Weight = ', w0_new, w_new)
w_lda = np.hstack((w_new,w0_new))
ax = plot_data.plot_data(X,y)
plot_data.plot_lines( X, y, w_lda, ax, color = 'black', label = 'LDA_alp', linestyle = '-')


ayak = SVM(C=1e6)
ayak.fit(X,y)
ayak.w
ayak.b
w_svm_sgo = np.hstack((ayak.w,ayak.b))
plot_data.plot_lines( X, y, w_svm_sgo, ax, color = 'black', label = 'SVM_sgo', linestyle = '-')
