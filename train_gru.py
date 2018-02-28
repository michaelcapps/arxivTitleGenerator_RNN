# train_gru.py
# Credit to this structure goes to Martin Gorner
# https://github.com/martin-gorner/tensorflow-rnn-shakespeare
#
# I used his shakespeare generator as a starting point and tutorial


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
batch_size = 200
time_steps = 30
state_size = 512
num_layers = 3
#learn_rate_start = 0.1
dropout_keep = 0.8

# Load data
file_name = 'arXivmathTitles_twoYears'
raw_text, num_titles = tx.read_data(file_name)
epoch_size = len(raw_text)//(batch_size*time_steps)

print("Text length: ", len(raw_text))
print("Epoch size: ", epoch_size)
print("Training on %i titles" %num_titles)

# Set up the model
#global_step = tf.Variable(0, trainable=False)
#decay_steps = 1.0
#decay_rate = 0.5
#learn_rate = tf.train.inverse_time_decay(learn_rate_start, global_step, decay_steps,decay_rate)
learn_rate = 0.001
pkeep = tf.placeholder(tf.float32,name = 'pkeep')  # dropout parameter
batchsize = tf.placeholder(tf.int32,name = 'batchsize')
lr = tf.placeholder(tf.float32,name = 'lr')

# One hot encoded inputs and outputs
X = tf.placeholder(tf.uint8, [None, None],name = 'X')    # batch_size,time_steps
Xo = tf.one_hot(X, NUM_CHARS, 1.0, 0.0)       # batch_size,time_steps, NUM_CHARS
Y_ = tf.placeholder(tf.uint8, [None, None],name = 'Y_')   # batch_size,time_steps
Yo_ = tf.one_hot(Y_, NUM_CHARS, 1.0, 0.0)     # batch_size,time_steps, NUM_CHARS


# Set up LSTM or GRU cells with dropout
cells = [rnn.GRUCell(state_size) for _ in range(num_layers)]
dropcells = [rnn.DropoutWrapper(cell,input_keep_prob=pkeep) for cell in cells]
stacked_cells = rnn.MultiRNNCell(dropcells,state_is_tuple=False)
stacked_cells = rnn.DropoutWrapper(stacked_cells, output_keep_prob=pkeep)

# Let dynamic_rnn do all the work
in_state = tf.placeholder(tf.float32, [None, state_size*num_layers],name = 'in_state')
Y_out, new_states = tf.nn.dynamic_rnn(stacked_cells, Xo, dtype=tf.float32, initial_state=in_state)
new_states = tf.identity(new_states,name = 'new_states')

# Flatten and set up softmax layer
Y_flat = tf.reshape(Y_out, [-1, state_size])    # batch_size*time_steps,state_size
Ylogits = layers.linear(Y_flat, NUM_CHARS)     # batch_size*time_steps,NUM_CHARS
Yflat_ = tf.reshape(Yo_, [-1, NUM_CHARS])     # batch_size*time_steps,NUM_CHARS
Yo = tf.nn.softmax(Ylogits,name = 'Yo')        # batch_size*time_steps,NUM_CHARS
Y = tf.argmax(Yo, 1)                          # batch_size*time_steps
Y = tf.reshape(Y, [batchsize, -1],name = 'Y')  # batch_size,time_steps

# Define our loss function
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)  # batch_size*time_steps
loss = tf.reshape(loss, [batchsize, -1])      # batch_size,time_steps

# Define the training step using AdamOptimizer
train_step = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss)

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
prev_epoch = 0

for x, y_, epoch in tx.batch_data_generator(raw_text, batch_size, time_steps, num_epochs):
    #print(epoch)
    if epoch != prev_epoch: #Starting new epoch
        #generate some titles and checkpoint
        print("End of epoch ", epoch - 1, ", generating some titles.")
        ry = np.array([[tx.char_to_code("K")]])
        rh = np.zeros([1, state_size * num_layers])
        for k in range(500):
            ryo, rh = sess.run([Yo, new_states], feed_dict={X: ry, pkeep: 1.0, in_state: rh, batchsize: 1})
            rc = tx.sample_probs(ryo, 3)
            print(tx.code_to_char(rc), end="")
            ry = np.array([[rc]])
        print("Done generating titles for this epoch.\n")
        saved_file = saver.save(sess, 'checkpoints/rnn_train', epoch)
        print("Saved file: " + saved_file)

    # train on one minibatch
    #feed_dict = {X: x, Y_: y_, in_state: istate, lr: learn_rate.eval(session = sess), pkeep: dropout_keep, batchsize: batch_size}
    feed_dict = {X: x, Y_: y_, in_state: istate, lr: learn_rate, pkeep: dropout_keep, batchsize: batch_size}
    _, y, ostate = sess.run([train_step, Y, new_states], feed_dict=feed_dict)

    istate = ostate

    step = step+1
    prev_epoch = epoch

saved_file = saver.save(sess, 'checkpoints/rnn_train_final')
print("Saved file: " + saved_file)



