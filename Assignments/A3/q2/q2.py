# Support Vector Machine (SVM) and Linear Discriminant Analysis (LDA)
# Importing the libraries
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
    plt.title('Data Points')
    plt.grid(linewidth=0.1)
    ax = plt.gca()
    return(ax)

  def plot_lines(self, X, y, w, ax, label, color = 'black', linestyle = '-', linewidth = 2, alpha = 1):
    color = color
    label = label
    linestyle = linestyle
    linewidth = linewidth
    ax.plot([X.min()-0.1, X.max()+0.1],[-(w[0]*(X.min()-0.1)+w[2])/w[1], -(w[0]*(X.max()+0.1)+w[2])/w[1]],
             label = label, linestyle = linestyle, linewidth = linewidth, color = color, alpha = alpha)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, title = 'Method')
#    plt.pause(pause_time)
    plt.show()

  def plot_error(self, errors, eta):
    plt.figure()
    plt.plot(errors,'.')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Trend when Learning Rate = '+ str(eta))
    plt.grid(lw = 0.2)

def plot_svc_decision_function(model, label ,ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=200, linewidth=1, facecolors='None', edgecolors = 'black');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.title('Data Points, Decision Boundary and Support Vectors')

def plot_lda_decision_function(model, label ,ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.predict_proba(xy)[:,1].reshape(X.shape)
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[0.5], alpha=0.5,
               linestyles=['-'])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.title('Data Points, Decision Boundary and Support Vectors')

X = np.array([
  [1., 3.],
  [2., 3.],
  [2., 4.],
  [3., 1.],
  [3., 2.],
  [4., 2.],
])

y = np.array([1,1,1,-1,-1,-1])


# AND
X = np.array([
  [0.0, 0.0],
  [0.0, 1.0],
  [1.0, 0.0],
  [1.0, 1.0],
])

y = np.array([-1,1,1,1])


# OR
X = np.array([
  [0.0, 0.0],
  [0.0, 1.0],
  [1.0, 0.0],
  [1.0, 1.0],
])

y = np.array([-1,-1,-1,1])

# Example
X = np.array([[1,1],
              [2,2],
              [4,4],
              [5,5]])

y = np.array([-1, -1, 1, 1])

# Example 2
X = np.array([
  [-2., 4.],
  [4, 1],
  [1, 6],
  [2, 4],
  [6, 2]
  ])

y = np.array([-1,-1,1,1,1])

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

# Fitting SVM to the Training set
from sklearn.svm import SVC
model = SVC(kernel = 'linear', random_state = 0, C = 1e6)
model.fit(X, y)
model.predict(X)
model.coef_
model.intercept_

# Plot the 2D data
plot_data = Plotter()
ax = plot_data.plot_data(X,y)
# Plot the Decision boundary and Sooprt Vectors
plot_svc_decision_function(model, label = 'SVM', ax = ax)
w_svm = np.hstack((model.coef_[0],model.intercept_))
plot_data.plot_lines(X,y, w_svm, ax, label = 'SVM', linestyle = 'dashed')


# Applying LinearSVC
from sklearn.svm import LinearSVC
linearSVC = LinearSVC(C = 1e6)
linearSVC.fit(X,y)
linearSVC.coef_
linearSVC.intercept_
w_l_SVC = np.hstack((linearSVC.coef_[0],linearSVC.intercept_))
plot_data.plot_lines( X, y, w_l_SVC, ax, color = 'cyan', label = 'Lin SVC', linestyle = 'dashed')

# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
for solver, color, linestyle in zip(['svd', 'eigen', 'lsqr'], ['red', 'green', 'blue'], ['dotted', 'dotted', 'dotted']):
  if (solver == 'svd'):
    lda = LDA(solver = solver, store_covariance=True)
  else:
    lda = LDA(solver = solver)
  lda.fit(X, y)
  lda.coef_
  lda.intercept_
  w_lda = np.hstack((lda.coef_[0],lda.intercept_))
  plot_data.plot_lines( X, y, w_lda, ax, color = color, label = 'LDA_' + solver, linestyle = linestyle)


plot_lda_decision_function(lda, label = 'lda')
lda.predict_proba(X)
