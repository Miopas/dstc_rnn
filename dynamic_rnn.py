''' 
A Dynamic Recurrent Neural Network (LSTM) implementation using
TensorFlow library. 
I modified the example offered in Tenserflow-Example for the DSTC2 task.

Project: https://github.com/aymericdamien/TensorFlow-Examples/
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import random
import pickle
import numpy as np
import pdb
import sys
from dstc_data import DstcData 

# ==========
#   MODEL
# ==========

if (len(sys.argv) < 6):
    print ('arg err, usage [python dynamic_rn.py slot_name train_iters learning_rate batch_size model_name]')
    exit()

# Parameters
slot_name = sys.argv[1]
training_iters = int(sys.argv[2])
learning_rate = float(sys.argv[3])
batch_size = int(sys.argv[4])
model_name = sys.argv[5]
display_step = 10

# Network Parameters
n_hidden = 64 # hidden layer num of features

# Get data
trainset = DstcData(slot_name, one_hot=False, multi_label=False, is_encode=False) 

n_input = trainset.max_turn_len
seq_max_len = trainset.max_n_steps # Sequence max length
n_classes = trainset.n_classes # linear sequence or not
print ("Model for slot [%s]\nn_classes is [%d]" % (slot_name, n_classes))
print ("n_input is [%d]\nseq_max_len is [%d]" % (n_input, seq_max_len))

# tf Graph input
x = tf.placeholder("float", [None, seq_max_len, n_input])
y = tf.placeholder("float", [None, n_classes])

# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def dynamicRNN(x, seqlen, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, seq_max_len, 0)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']

pred = dynamicRNN(x, seqlen, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# for save models
saver = tf.train.Saver()

# Launch the graph
config = tf.ConfigProto(device_count={"CPU": 4}, # limit to num_cpu_core CPU usage  
        inter_op_parallelism_threads = 1,   
        intra_op_parallelism_threads = 4,  
        log_device_placement=True) 
with tf.Session(config = config) as sess:
    sess.run(init)
    step = 1

    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y, batch_seqlen = trainset.next_batch(batch_size)

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       seqlen: batch_seqlen})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,
                                                seqlen: batch_seqlen})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y,
                                             seqlen: batch_seqlen})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    saver.save(sess, model_name + ".tfmodel")
    print("Optimization Finished!")
