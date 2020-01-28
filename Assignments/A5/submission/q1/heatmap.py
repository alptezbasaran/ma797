# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 00:33:17 2019

@author: atezbas
"""

import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

df = pd.read_pickle('rnn_metrics.pkl')
data = df.to_numpy()
data = np.hstack((data[:,0:2],data[:,-1].reshape(12,1)))

test_error = data[:,-1].reshape(4,3)

fig, ax = plt.subplots()
xlabels = [8,16,32]
ylabels = [16,32,64,128]
ax = sns.heatmap(test_error, annot=True, cmap = 'coolwarm', xticklabels=xlabels, yticklabels=ylabels, vmin=np.min(test_error), vmax=np.max(test_error), fmt='.5f')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_ylabel('Hidden States')
ax.set_xlabel('Vector Length')
