# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 12:54:11 2019

@author: atezbas
"""

import numpy as np
import matplotlib.pyplot as plt


class Plotter:

  def plot_data(self, X, y):
    plt.figure()
    plt.subplot(111)
    plt.subplots_adjust(right = 0.8)
    plt.ion()
    marker = []
    color = []
    for i in range(len(y)):
      marker.append('+' if y[i] == 1 else '_')
      color.append('blue' if y[i] == 1 else 'red')
      plt.scatter(X[i,0],X[i,1], marker = marker[i] , s = 120, c = color[i])
    plt.xlim([X[:,0].min()-0.1, X[:,0].max()+0.1])
    plt.ylim([X[:,1].min()-0.1, X[:,1].max()+0.1])
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Perceptron Iterations')
    plt.grid(linewidth=0.1)
    ax = plt.gca()
    return(ax)

  def plot_lines(self, t, X, y, w, ax, epochs ,pause_time = 0.1):
    if t == 0:
      label = str(t)
      linestyle = ':'
      linewidth = 2
    elif t > epochs - 15:
      label = str(t)
      linestyle = '-'
      linewidth = 1
    else:
      label = None
      linestyle = '-'
      linewidth = 1
    ax.plot([X.min()-0.1, X.max()+0.1],[-(w[0]*(X.min()-0.1)+w[2])/w[1], -(w[0]*(X.max()+0.1)+w[2])/w[1]],
             lw = 0.5, label = label, linestyle = linestyle, linewidth = linewidth)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, title = 'Epoch')
    plt.pause(pause_time)
    plt.show()

  def plot_error(self, errors, eta):
    plt.figure()
    plt.plot(errors,'.')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Trend when Learning Rate = '+ str(eta))
    plt.grid(lw = 0.2)

class Perceptron:

  import numpy as np

  def __init__(self, X, y, plot_data_lines = False, plot_errors = False):

    X = np.insert(X, X.shape[1], 1, axis = 1) # Insert 1s to 0th column
    self.X = X
    self.y = y.reshape(len(y),1)
    self.plot_data_lines  = plot_data_lines
    self.plot_errors = plot_errors

  def predict(self, X):
    if X.shape[1] != self.w.shape[0]: X = np.insert(X, X.shape[1], 1, axis = 1) # Insert 1s to 0th column
    predicted = np.sign(np.dot(X,self.w))
    predicted = predicted[:, np.newaxis]
    return(predicted)

  def error(self,y_pred,y_test):
    y_test= y_test.reshape(len(y_test),1)
    print(y_test.shape)
    print(y_pred.shape)
    error = float(np.count_nonzero(y_pred-y_test)) / float(len(y_test))
    accuracy = 1 - error
    return(error, accuracy)

  def train(self, w, eta = 1, epochs = 20):
    self.w = w
    if self.plot_data_lines or self.plot_errors:
      new_figure = Plotter()
      if self.plot_data_lines:
        ax = new_figure.plot_data(self.X, self.y)
        new_figure.plot_lines(0, self.X, self.y ,self.w,ax , epochs)

    accuracy = []

    for t in range(epochs):
      for i, x in enumerate(self.X):
        if (np.dot(self.X[i], self.w)*self.y[i]) <= 0:
          self.w = self.w + eta*self.X[i]*self.y[i]
      if self.plot_data_lines: new_figure.plot_lines(t+1,self.X, self.y, self.w,ax, epochs)
      y_pred = self.predict(self.X)
      accuracy.append(1-float(np.count_nonzero(y_pred-self.y)) / float(len(self.y)))
    if self.plot_errors: new_figure.plot_error(accuracy, eta)
    return(self.w)

# Use each data one by one
import scipy.io as sio


# Question 1

X = np.array([
  [0.0, 1.5],
  [1.0, 1.0],
  [2.0, 2.0],
  [2.0, 0.0],
  [0.0, 0.0],
  [1.0, 0.0],
  [0.0, 1.0],
])

y = np.array([1,1,1,1,-1,-1,-1]).reshape(7,1)
w_init = np.array([4, 1, -2])

perceptron = Perceptron(X, y, plot_errors = True, plot_data_lines = True)
w_last = perceptron.train(w_init, epochs = 10)
print('Ayak weights = ', w_last)
y_pred = perceptron.predict(X)
perceptron.error(y_pred, y)

# Question 2

data = sio.loadmat('SampleCredit.mat')
X_train = data['sample'][0:500,]
X_test  = data['sample'][500:,]
y_train = data['label'][0:500,0].reshape(500,1)
y_test  = data['label'][500:,0].reshape(X_test.shape[0],1)


# Correct Normalized
for i in range(X_train.shape[1]):
  X_train[:,i] = np.divide(X_train[:,i],max(X_train[:,i]))
  X_test[:,i] = np.divide(X_test[:,i],max(X_test[:,i]))

w = np.zeros(len(X_train[0])+1)
#  w = np.random.randn(len(X_train[0])+1)
big_perceptron = Perceptron(X_train,y_train, plot_errors= True)
w_big1 = big_perceptron.train(w, eta = 0.1, epochs = 1000)
w_big2 = big_perceptron.train(w_big1, eta = 0.00001, epochs = 1000)
w_big3 = big_perceptron.train(w_big2, eta = 0.0040, epochs = 1000)
w_big4 = big_perceptron.train(w_big3, eta = 0.0001, epochs = 1000)
w_big5 = big_perceptron.train(w_big4, eta = 0.00001, epochs = 1000)
print(w_big5)
#  print('Buyuk ayak weights = ', w_big)
y_pred = big_perceptron.predict(X_test)
[error_test, accuracy_test] = big_perceptron.error(y_pred,y_test)
y_pred_train = big_perceptron.predict(X_train)
[error_train, accuracy_train] = big_perceptron.error(y_pred_train,y_train)
 # Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_pred = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_pred_train)

# AND
X = np.array([
  [0.0, 0.0],
  [0.0, 1.0],
  [1.0, 0.0],
  [1.0, 1.0],
])

y = np.array([-1,1,1,1]).reshape(4,1)
w = np.array([-2, 2, 0])
andPerceptron = Perceptron(X,y,plot_data_lines = True, plot_errors = True)
w_and = andPerceptron.train(w,epochs = 10)
print(w_and)

# OR
X = np.array([
  [0.0, 0.0],
  [0.0, 1.0],
  [1.0, 0.0],
  [1.0, 1.0],
])

y = np.array([-1,-1,-1,1]).reshape(4,1)
w = np.array([4, 1, -2])
orPerceptron = Perceptron(X,y,plot_data_lines = True, plot_errors = True)
w_or = orPerceptron.train(w,epochs = 10)
print(w_or)

# Example 1
X = np.array([
  [-2., 4.],
  [4, 1],
  [1, 6],
  [2, 4],
  [6, 2]
  ])

y = np.array([-1,-1,1,1,1])
w = np.array([6, 3, -1])
ExamplePerceptron = Perceptron(X,y,plot_data_lines = True, plot_errors = True)
w_ex = ExamplePerceptron.train(w,epochs = 30)
print(w_ex)

# Example 2
X = np.array([[1,1],
              [2,2],
              [4,4],
              [5,5]])
y = np.array([-1, -1, 1, 1])
w = np.array([0,99, 5])
Example2Perceptron = Perceptron(X,y,plot_data_lines = True, plot_errors = True)
w_ex2 = Example2Perceptron.train(w,epochs = 20)
print(w_ex2)



from sklearn.linear_model import Perceptron
sk_perceptron = Perceptron(tol=1e-5, random_state=0)
sk_perceptron.fit(X,y)
print(sk_perceptron.score(X,y))
print(sk_perceptron.get_params())
print([sk_perceptron.coef_, sk_perceptron.intercept_])
print(sk_perceptron.n_iter_)

sk_bigdata = Perceptron(max_iter = 1000, eta0=0.1, tol=1e-5, random_state=0)
sk_bigdata.fit(X_train, y_train)
print([sk_bigdata.coef_, sk_bigdata.intercept_])
print('Accuracy Training= ',sk_bigdata.score(X_train, y_train)*100)
print('Accuracy Testing= ',sk_bigdata.score(X_test, y_test)*100)


ciplakAyak = X-np.mean(X, axis = 0)
cov = np.dot(ciplakAyak.T,ciplakAyak)/X.shape[0]
print(cov)
cov_numpy = np.cov(X, rowvar = False, ddof = 0)
print(cov_numpy)
