

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
from tensorflow.contrib import ctc



# Global parameters
learning_rate = 0.001
learning_iterations = 100  # number of mini-batches
mini_batch_size = 100
n_hidden_layer = 128
n_output_classes = 28  # Number of letters in our alphabet


# Input Data Parameters
batch_sizes = [(5, 10)]  # (i, o) Divide input into batches of max input size of i timesteps and max output label size o. If < i or < o should be padded accordingly
n_input = 8  # Number of input features for each sliding window of 1px

number_of_timesteps = tf.Variable(0)

# Network weights. Does not depend on batch size
hidden_weights = tf.Variable(tf.random_normal([n_input, 2 * n_hidden_layer]))  # 2 * n_hidden_layer for forward and backward layer
hidden_biases = tf.Variable(tf.random_normal([2 * n_hidden_layer]))

output_weights = tf.Variable(tf.random_normal([2 * n_hidden_layer, n_output_classes]))
output_biases = tf.Variable(tf.random_normal([n_output_classes]))


# Tensorflow graph inout/output
x = tf.placeholder("float", [None, None, n_input])
x_length = tf.placeholder("int32", [None])
y = tf.placeholder("int32", [None, n_output_classes])

istate_fw = tf.placeholder("float", [None, 2 * n_hidden_layer])
istate_bw = tf.placeholder("float", [None, 2 * n_hidden_layer])


# Adapted for different length of input batches code from brnn minst tutorial.
# Used to calculate forward pass of network
# n_steps - number of timesteps in current batch
def blstm_layer(_X, n_steps, _istate_fw, _istate_bw):

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
    _X = tf.split(0, n_steps, _X)  # n_steps * (batch_size, n_hidden)

    # Get lstm cell output
    outputs = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, _X,
                                            initial_state_fw=_istate_fw,
                                            initial_state_bw=_istate_bw)

    # Linear activation
    # Get inner loop last output
    # TODO Should be all outputs? Not only last ones
    return tf.matmul(outputs[-1], output_weights) + output_biases

prediction = blstm_layer(x, number_of_timesteps, istate_fw, istate_bw)

cost = ctc.ctc_loss(prediction, y, x_length)

def get_batch(max_input, max_ouput, max_batch_size):
    # Placeholder function. Should be defined in separate file
    batch_x = None
    batch_y = None
    return batch_x, batch_y


def process_batch(max_input, max_ouput, batch_size=mini_batch_size):
    xs, ys = get_batch(max_input, max_ouput, batch_size)
    real_batch_size = len(xs)
    xs.reshape((real_batch_size, max_input, n_output_classes))

    

