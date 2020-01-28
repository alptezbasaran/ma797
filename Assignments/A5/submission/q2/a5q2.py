# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:00:04 2019

@author: atezbas

"""

import numpy as np
import matplotlib.pyplot as plt
from time import time

#Import Data
import scipy.io as sio
data = sio.loadmat('Pima-Data-Adjusted.mat')['data']

# Investigate the data
plt.figure()
for _ in range(data.shape[1]):
  mean = np.around(np.mean(data[:,_]), decimals = 2)
  var  = np.around(np.var(data[:,_]),  decimals = 2)
  print('+'*35)
  print('Mean of feature '+ str(_+1) + ' is ' + str(mean))
  print('Var. of feature '+ str(_+1) + ' is ' + str(var))
  index = int('33'+str(_+1))
  ax = plt.subplot(index)
  if _ == data.shape[1]-1:
    plt.title('Labels ')
  else:
    plt.title(str(_+1))
    plt.text(0.83, 0.9, r'$\mu=$' + str(mean), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize  = 8)
    plt.text(0.83, 0.78, r'$\sigma^2=$' + str(var), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize  = 8)
    ax.axvline(x = mean, linewidth = 1.2, color = 'red')
  plt.hist(data[:,_], bins=20, edgecolor='black', linewidth=0.6, alpha=0.8)
  plt.tight_layout()

# Features and Labels
X = data[:,0:-1]
y = data[:,-1]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

# Decision Tree
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(random_state = 0)
#dt_classifier.fit(X_train, y_train)
## Predicting the Test set results
#y_pred = dt_classifier.predict(X_test)
## Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm_dt = confusion_matrix(y_test, y_pred)

# Random Forest
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state = 0)
#rf_classifier.fit(X_train, y_train)
## Predicting the Test set results
#y_pred = rf_classifier.predict(X_test)
## Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm_rf = confusion_matrix(y_test, y_pred)


# Grid search with k-fold cross validation
from sklearn.model_selection import GridSearchCV
# use a full grid over all parameters
param_grid = {"max_depth": [2, 3, 4, 5, 6, 7, 8,9, 10, 11, 20, 50, None],
              "min_samples_split": [2, 3, 4, 5, 10, 20]}
# run grid search
grid_search = GridSearchCV(dt_classifier, param_grid=param_grid, cv=5, iid = False)
start = time()
grid_search.fit(X, y)
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
print(grid_search.best_params_)