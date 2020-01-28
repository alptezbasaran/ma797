# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 17:42:40 2019

@author: atezbas
"""

import tensorflow as tf
from tensorflow.keras.datasets import imdb
import pandas as pd
import shutil
import os

cwd = os.getcwd()
shutil.rmtree(cwd + '\\rnn_*',ignore_errors=True)

tf.__version__
# Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
tf.random.set_seed(0)

# Data import parameters
number_of_words = 5000
max_len = 500

# Import data
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = number_of_words)

# Padding all sequences to be the same lenght
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, max_len)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, max_len)

# Define model architecture
def create_RNN_model(hidden_states = 64, vector_length = 16):
  # Create the model
  model = tf.keras.Sequential()
  # 1st Embedding layer
  model.add(tf.keras.layers.Embedding(input_dim = number_of_words, output_dim = vector_length, input_shape = (X_train.shape[1],)))
  # 1st LSTM layer
  model.add(tf.keras.layers.LSTM(units = hidden_states, activation = 'tanh'))
  # Output layer
  model.add(tf.keras.layers.Dense(units = 1, activation ='sigmoid'))
  # Compile
  model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['binary_accuracy'])
  return(model)

# Iterate over parameters
vector_length = [8,16,32]
hidden_states = [16,32,64,128]

print('Training all')
metrics = []
for vl in vector_length:
  for hs in hidden_states:
    print('+'*50)
    print('vector length = ', vl, 'hidden states = ',hs)
    print('+'*50)
    # Tensorboard
    log_dir = 'rnn_log_hs'+ str(hs) + '_vec_len_' + str(vl)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch', profile_batch = 0)
    # Early Stop
    early_stoppping = tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', verbose=1)
    # Save
    save_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath= 'rnn_log_hs'+ str(hs) + '_vec_len_' + str(vl) + '.h5',
                                                         save_best_only=True,
                                                         monitor='val_loss',
                                                         verbose=1)
    # Create Model
    model = create_RNN_model(vector_length = vl, hidden_states = hs)
    # Fit
    history = model.fit(X_train, y_train, epochs = 20,  validation_split = 0.2, use_multiprocessing = True,
                                                                   verbose = 1, callbacks = [tensorboard_callback, early_stoppping, save_checkpoint])
    # Metrics
    training_loss = history.history['loss'][-1]
    validation_loss = history.history['val_loss'][-1]
    training_accuracy = history.history['binary_accuracy'][-1]
    validation_accuracy = history.history['val_binary_accuracy'][-1]
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose = 2)
    print('Test accuracy = ',test_accuracy)
    metrics.append({'hidden_state':hs,
                    'vector_length':vl,
                    'validation_accuracy':validation_accuracy,
                    'training_loss':training_loss,
                    'validation_loss':validation_loss,
                    'training_accuracy':training_accuracy,
                    'test_loss':test_loss,
                    'test_accuracy':test_accuracy
                    })
print('Training Over')
df = pd.DataFrame(metrics)
df.to_pickle('./rnn_metrics.pkl')
print('Database ready')