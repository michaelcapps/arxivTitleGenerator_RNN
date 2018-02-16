# train.py

import os
import math
import numpy as np
import text_handler as tx
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Set parameters
NUM_CHARS = tx.NUM_CHARS
num_epochs = 10
batch_size = 100
time_steps = 40
state_size = 512
num_layers = 2
learn_rate_start = 0.1
dropout_keep = 0.8
cell_type = 'lstm'
#cell_type = 'gru'

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
learn_rate = tf.train.inverse_time_decay(learn_rate_start, global_step, decay_steps,decay_rate)
pkeep = tf.placeholder(tf.float32)  # dropout parameter
batchsize = tf.placeholder(tf.int32)
lr = tf.placeholder(tf.float32)

# One hot encoded inputs and outputs
X = tf.placeholder(tf.uint8, [None, None])    # batch_size,time_steps
Xo = tf.one_hot(X, NUM_CHARS, 1.0, 0.0)       # batch_size,time_steps, NUM_CHARS
Y_ = tf.placeholder(tf.uint8, [None, None])   # batch_size,time_steps
Yo_ = tf.one_hot(Y_, NUM_CHARS, 1.0, 0.0)     # batch_size,time_steps, NUM_CHARS
in_state = tf.placeholder(tf.float32, [None, state_size*num_layers]) 

# Set up LSTM or GRU cells with dropout
if cell_type == 'gru':
	cells = [rnn.GRUCell(state_size) for j in range(num_layers)]
elif cell_type == 'lstm':
	cells = [rnn.LSTMCell(state_size,state_is_tuple=True) for _ in range(num_layers)]
dropcells = [rnn.DropoutWrapper(cell,input_keep_prob=pkeep) for cell in cells]
stacked_cells = rnn.MultiRNNCell(dropcells, state_is_tuple=True)
stacked_cells = rnn.DropoutWrapper(stacked_cells, output_keep_prob=pkeep)

# Let dynamic_rnn do all the work
init_state = stacked_cells.zero_state(batchsize,tf.float32)
Y_out, last_state = tf.nn.dynamic_rnn(stacked_cells, Xo, dtype=tf.float32, initial_state=init_state)

# Flatten and set up softmax layer
Y_flat = tf.reshape(Y_out, [-1, state_size])    # batch_size*time_steps,state_size
Ylogits = layers.linear(Y_flat, NUM_CHARS)     # batch_size*time_steps,NUM_CHARS
Yflat_ = tf.reshape(Yo_, [-1, NUM_CHARS])     # batch_size*time_steps,NUM_CHARS
Yo = tf.nn.softmax(Ylogits)        # batch_size*time_steps,NUM_CHARS
Y = tf.argmax(Yo, 1)                          # batch_size*time_steps
Y = tf.reshape(Y, [batchsize, -1])  # batch_size,time_steps

# Define our loss function
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)  # batch_size*time_steps
loss = tf.reshape(loss, [batchsize, -1])      # batch_size,time_steps

# Define the training step using AdamOptimizer
train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)

# Define saver to create checkpoints during training
if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
saver = tf.train.Saver()

# initialize variables, create tf session
istate = np.zeros([batch_size, state_size*num_layers])
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
step = 0

for x, y_, epoch in tx.batch_data_generator(raw_text, batch_size, time_steps, 1):

    # train on one minibatch
    feed_dict = {X: x, Y_: y_, in_state: istate, lr: learn_rate, pkeep: dropout_keep, batchsize: batch_size}
    _, y, ostate = sess.run([train_step, Y, last_state], feed_dict=feed_dict)

    istate = ostate
    step = step + batch_size + time_steps

saved_file = saver.save(sess, 'checkpoints/rnn_train_' + timestamp, global_step=step)
print("Saved file: " + saved_file)
