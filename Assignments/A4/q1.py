# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:35:01 2019

@author: atezbas
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime

tf.__version__
tf.random.set_seed(0)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Look at images
plt.imshow(x_test[0])

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],x_test.shape[2],1))
#plt.imshow(x_test[0])

#Keras sequential
model = tf.keras.models.Sequential()
# 1. Conv2D
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding="same", activation="relu", input_shape=(28,28,1)))
# 2. Maxpool
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'))
# 3. Conv2D
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
# 4. Maxpool
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'))
# 5. Flatten
model.add(tf.keras.layers.Flatten())
# 6. Dense
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
# 7. Dropout
model.add(tf.keras.layers.Dropout(0.5))
# 8. Output
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

# Summary
model.summary()

# Compile
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="Adam", metrics=["sparse_categorical_accuracy"])

# Tensorboard
log_dir = 'log'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Fit
model.fit(x_train, y_train,batch_size = 50, validation_split = 0.2 ,epochs=5000, callbacks=[tensorboard_callback])

# Eval
test_loss, test_accuracy = model.evaluate(x_test, y_test)

for i in range(x_train.shape[0]):
  plt.imshow(x_test[i,:,:,0])
