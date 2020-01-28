# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:58:07 2019

@author: atezbas
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt

#Importing with custom names to avoid issues with numpy / sympy matrix
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers


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

#Initializing values and computing H. Note the 1. to force to float type
m,n = X.shape
y = y.reshape(-1,1) * 1.
X_dash = y * X
H = np.dot(X_dash , X_dash.T) * 1.      # Q, done

X
H
X_dash
y

#Converting into cvxopt format
P = cvxopt_matrix(H)                    # Q, done
q = cvxopt_matrix(-np.ones((m, 1)))
G = cvxopt_matrix(-np.eye(m))
h = cvxopt_matrix(np.zeros(m))
A = cvxopt_matrix(y.reshape(1, -1))
b = cvxopt_matrix(np.zeros(1))

#Setting solver parameters (change default to decrease tolerance)
cvxopt_solvers.options['show_progress'] = True
cvxopt_solvers.options['abstol'] = 1e-10
cvxopt_solvers.options['reltol'] = 1e-10
cvxopt_solvers.options['feastol'] = 1e-10

#Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])

alphas
#w parameter in vectorized form
w = ((y * alphas).T @ X).reshape(-1,1)

#Selecting the set of indices S corresponding to non zero parameters
S = (alphas > 1e-4).flatten()

#Computing b
b = y[S] - np.dot(X[S], w)

#Display results
print('Alphas = ',alphas[alphas > 1e-4])
print('w = ', w.flatten())
print('b = ', b[0])


y_dot = np.dot(y,y.T)
X_dot = np.dot(X,X.T)

Q = y_dot * X_dot
Q
H
H == Q

q = np.ones((X.shape[0],1))
q

w_svm = np.hstack((w.flatten(),b[0]))
w_svm

plot_data.plot_lines(X,y, w_svm, ax, label = 'SVM', color = 'green', linestyle = 'dashed')
