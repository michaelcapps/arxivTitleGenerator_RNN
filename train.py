# This trains a RNN on the title data retrieved from the arXiv

import tensorflow as tf 
import pandas as pd 
import numpy as np
from sklearn.utils import shuffle


# Preprocess the data
with open('arXivTitles','r') as f:
	text = f.read()
	chars = sorted(list(set(text))) # for index mapping
	char_indices = dict((c, i) for i, c in enumerate(chars))
	indices_char = dict((i, c) for i, c in enumerate(chars))
	print('Num chars: ',len(chars))
	f.seek(0)
	titles = f.readlines()
	titles = titles[1:] #remove column header
	print('Num titles: ',len(titles))

# Store each title as a one hot encoded batch, labels are the next character
x = [] #data
y = [] #labels
for title in titles:
	X = np.zeros((len(title),len(chars)),dtype = np.bool)
	for i, char in enumerate(title):
		X[i,char_indices[char]]=1
	x.append(X)
	y.append(np.roll(X,-1,axis = 0))

# Model parameters
num_epochs = len(x)
trunc_backprop_len = 5
state_size = 4
num_classes = len(chars)


