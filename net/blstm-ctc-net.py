import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
from tensorflow.contrib import ctc

import numpy as np
import datetime

import rnn_mnist.mnist_input_data as mnist_input_data

mnist = mnist_input_data.read_data_sets("/tmp/data/", one_hot=True)

# Global parameters
learning_rate = 0.001
learning_iterations = 100  # number of mini-batches
mini_batch_size = 100
n_hidden_layer = 128
n_output_classes = 28  # Number of letters in our alphabet
display_time = 5

# Input Data Parameters
batch_size = 128  # (i, o) Divide input into batches of max input size of i timesteps and max output label size o. If < i or < o should be padded accordingly
n_input = 28  # Number of input features for each sliding window of 1px
max_input_timesteps = 28

# Network weights. Does not depend on batch size
hidden_weights = tf.Variable(tf.random_normal([n_input, 2 * n_hidden_layer]))  # 2 * n_hidden_layer for forward and backward layer
hidden_biases = tf.Variable(tf.random_normal([2 * n_hidden_layer]))

output_weights = tf.Variable(tf.random_normal([2 * n_hidden_layer, n_output_classes]))
output_biases = tf.Variable(tf.random_normal([n_output_classes]))

# Tensorflow graph inout/output
x = tf.placeholder("float", [batch_size, max_input_timesteps, n_input])
x_length = tf.placeholder("int32", [batch_size])

y_index = tf.placeholder("int64", [None, 2])
y_labels = tf.placeholder("int32", [None])

istate_fw = tf.placeholder("float", [None, 2 * n_hidden_layer])
istate_bw = tf.placeholder("float", [None, 2 * n_hidden_layer])


# Adapted for different length of input batches code from brnn minst tutorial.
# Used to calculate forward pass of network
# n_steps - number of timesteps in current batch
def blstm_layer(_X, _istate_fw, _istate_bw, _x_length):
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input])  # (n_steps*batch_size, n_input)
    # Linear activation
    _X = tf.matmul(_X, hidden_weights) + hidden_biases

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn_cell.BasicLSTMCell(n_hidden_layer, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn_cell.BasicLSTMCell(n_hidden_layer, forget_bias=1.0)
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, max_input_timesteps, _X)  # n_steps * (batch_size, n_hidden)

    # Get lstm cell output
    outputs, output_state_fw, output_state_bw = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, _X,
                                                                      initial_state_fw=_istate_fw,
                                                                      initial_state_bw=_istate_bw,
                                                                      sequence_length=_x_length
                                                                      )

    outputs = tf.concat(0, outputs)
    activation = tf.matmul(outputs, output_weights) + output_biases

    return tf.reshape(activation, [max_input_timesteps, batch_size, n_output_classes])


prediction = blstm_layer(x, istate_fw, istate_bw, x_length)

cost = ctc.ctc_loss(prediction, tf.SparseTensor(indices=y_index, values=y_labels, shape=[batch_size, max_input_timesteps]), x_length)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    prev_output_time = datetime.datetime.now()
    # Keep training until reach max iterations
    while step < learning_iterations:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_xs = batch_xs.reshape((batch_size, max_input_timesteps, n_input))
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                       istate_fw: np.zeros((batch_size, 2 * n_hidden_layer)),
                                       istate_bw: np.zeros((batch_size, 2 * n_hidden_layer))})
        from_prev_output_time = datetime.datetime.now() - prev_output_time

        if step == 1 or from_prev_output_time.seconds > display_time:
           pass
        step += 1
    # print("Optimization Finished!")
    # # Calculate accuracy for 128 mnist test images
    # test_len = 128
    # test_data = mnist.test.images[:test_len].reshape((-1, max_input_timesteps, n_input))
    # test_label = mnist.test.labels[:test_len]
    # print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label,
    #                                                          istate_fw: np.zeros((test_len, 2 * n_hidden_layer)),
    #                                                          istate_bw: np.zeros((test_len, 2 * n_hidden_layer))}))
