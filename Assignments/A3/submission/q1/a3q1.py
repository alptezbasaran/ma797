# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import scipy.io as sio
import tensorflow as tf
import datetime

tf.__version__
tf.random.set_seed(0)

tf.keras.backend.clear_session
# A3 Q1 data

data = sio.loadmat('SampleCredit.mat')
X_train = data['sample'][0:500,]
X_test  = data['sample'][500:,]
y_train = data['label'][0:500,0].reshape(500,1)
y_test  = data['label'][500:,0].reshape(X_test.shape[0],1)

y_train[y_train==-1] = 0
y_test[y_test==-1] = 0


# Normalization
for i in range(X_train.shape[1]):
  X_train[:,i] = np.divide(X_train[:,i],max(X_train[:,i]))
  X_test[:,i] = np.divide(X_test[:,i],max(X_test[:,i]))

#Keras sequential
model = tf.keras.models.Sequential()
# The first layer
model.add(tf.keras.layers.Dense(units = 30, activation = 'linear', input_shape = (15, )))
#model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units = 30, activation = 'tanh'))
#model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units =  2, activation = 'softmax'))
# Compiling
model.compile(optimizer='adam' ,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Summary
model.summary()

# Tensorboard
log_dir = 'log'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Fit
model.fit(X_train, y_train, use_multiprocessing = True , validation_data = (X_test, y_test) ,epochs = 1000, callbacks=[tensorboard_callback])
# Eval
test_loss, test_accuracy = model.evaluate(X_test,y_test, verbose = 0)
test_loss, test_accuracy
