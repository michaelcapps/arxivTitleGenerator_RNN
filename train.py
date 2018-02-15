# train.py

import os
import math
import numpy as np
import text_handler as tx
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers

# Set parameters
NUM_CHARS = tx.NUM_CHARS
batch_size = 100
time_steps = 40
state_size = 512
num_layers = 2
learn_rate_start = 0.1

# Load data
file_name = 'arXivmathTitles'
raw_text, num_titles = tx.read_data(file_name)
epoch_size = len(raw_text)//(batch_size*time_steps)

print("Text length: ", len(raw_text))
print("Epoch size: ", epoch_size)
print("Training on %i titles" %num_titles)

# Set up the model
global_step = tf.Variable(0, trainable=False)
decay_steps = 1.0
decay_rate = 0.5
lr = tf.train.inverse_time_decay(learn_rate_start, global_step, decay_steps,decay_rate)
pkeep = tf.placeholder(tf.float32)  # dropout parameter
batchsize = tf.placeholder(tf.int32)

# inputs
X = tf.placeholder(tf.uint8, [None, None])    # batch_size,time_steps
Xo = tf.one_hot(X, NUM_CHARS, 1.0, 0.0)       # batch_size,time_steps, NUM_CHARS
# outputs
Y_ = tf.placeholder(tf.uint8, [None, None])   # batch_size,time_steps
Yo_ = tf.one_hot(Y_, NUM_CHARS, 1.0, 0.0)     # batch_size,time_steps, NUM_CHARS
# input state
in_state = tf.placeholder(tf.float32, [None, state_size*num_layers])  # batch_size, state_size*num_layers

# cells and dropout
gru_cells = [rnn.GRUCell(state_size) for j in range(num_layers)]
#lstm_cells = [rnn.LSTMCell(state_size) for j in range(num_layers)]
#dropcells = [rnn.DropoutWrapper(cell,input_keep_prob=pkeep) for cell in lstm_cells]
dropcells = [rnn.DropoutWrapper(cell,input_keep_prob=pkeep) for cell in gru_cells]
multicell = rnn.MultiRNNCell(dropcells, state_is_tuple=False)
multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)




