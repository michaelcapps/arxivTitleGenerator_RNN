# Credit to this structure goes to Martin Gorner
# https://github.com/martin-gorner/tensorflow-rnn-shakespeare
#
# I used his shakespeare generator as a starting point and tutorial

import tensorflow as tf
import numpy as np
import text_handler as tx

# these must match what was saved !
NUM_CHARS = tx.NUM_CHARS
num_layers = 3
state_size = 512

arxiv_checkpoint = "checkpoints/rnn_train_final.meta"

topn = 2

ncnt = 0
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(arxiv_checkpoint)
    new_saver.restore(sess, tf.train.latest_checkpoint('checkpoints/'))
    graph = sess.graph
    
    #new_saver.restore(sess, 'checkpoints/rnn_train_final')
    x = tx.char_to_code("L")
    x = np.array([[x]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1

    # initial values
    y = x
    h = np.zeros([1, state_size*num_layers], dtype=np.float32)  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]
    for i in range(10000):
        yo, h = sess.run(['Yo:0', 'new_states:0'], feed_dict={'X:0': y, 'pkeep:0': 1.0, 'in_state:0': h, 'batchsize:0': 1})
        c = tx.sample_probs(yo, topn)
        y = np.array([[c]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
        c = tx.code_to_char(c)
        print(c, end="")

        if c == '\n':
            ncnt = 0
        else:
            ncnt += 1
        if ncnt == 100:
            print("")
            ncnt = 0

