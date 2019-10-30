# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:07:44 2019

@author: atezbas
"""

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the data set
movies =  pd.read_csv('ml-1m/movies.dat', sep='::', header = None, engine = 'python', encoding = 'latin-1')
