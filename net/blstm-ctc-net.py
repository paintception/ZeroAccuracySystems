import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
from tensorflow.contrib import ctc

import numpy as np
import datetime

import net.word_dataset_with_timesteps as wd
import net.dirs as dirs

print("Loading data")
word_dataset = wd.WordDataSet(dir_path=dirs.STANFORD_PROCESSED_WORD_BOXES_DIR_PATH)

display_time = 10

# Global parameters
learning_rate = 0.001
learning_iterations = 100  # number of mini-batches
batch_size = 128  # number of words in a batch
n_hidden_layer = 128  # number of nodes in hidden layer
n_output_classes = len(word_dataset.unique_chars) + 1  # Number of letters in our alphabet and empty label

n_input = word_dataset.get_feature_count()  # Number of input features for each sliding window of 1px
max_input_timesteps = 100

# Network weights. Does not depend on batch size
hidden_weights = tf.Variable(tf.random_normal([n_input, 2 * n_hidden_layer]))  # 2 * n_hidden_layer for forward and backward layer
hidden_biases = tf.Variable(tf.random_normal([2 * n_hidden_layer]))

output_weights = tf.Variable(tf.random_normal([2 * n_hidden_layer, n_output_classes]))
output_biases = tf.Variable(tf.random_normal([n_output_classes]))

# Tensorflow graph inout/output
x = tf.placeholder("float", [None, max_input_timesteps, n_input])
x_length = tf.placeholder("int32", [None])

y_index = tf.placeholder("int64", [None, 2])
y_labels = tf.placeholder("int32", [None])


# Used to calculate forward pass of network
# n_steps - number of timesteps in current batch
def blstm_layer(_X, _x_length):
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input])  # (n_steps*batch_size, n_input)
    # Linear activation
    _X = tf.matmul(_X, hidden_weights) + hidden_biases

    # Forward direction cell
    lstm_fw_cell = rnn_cell.BasicLSTMCell(n_hidden_layer, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn_cell.BasicLSTMCell(n_hidden_layer, forget_bias=1.0)

    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, max_input_timesteps, _X)  # n_steps * (batch_size, n_hidden)

    istate_fw = lstm_fw_cell.zero_state(batch_size, tf.float32)
    istate_bw = lstm_bw_cell.zero_state(batch_size, tf.float32)

    # Get lstm cell output
    outputs, output_state_fw, output_state_bw = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, _X,
                                                                      initial_state_fw=istate_fw,
                                                                      initial_state_bw=istate_bw,
                                                                      sequence_length=_x_length
                                                                      )

    outputs = tf.concat(0, outputs)
    activation = tf.matmul(outputs, output_weights) + output_biases

    tf.Print(activation, [activation], message="Activation: ")
    return tf.reshape(activation, [max_input_timesteps, batch_size, n_output_classes])


prediction = blstm_layer(x, x_length)

cost = tf.reduce_mean(ctc.ctc_loss(prediction, tf.SparseTensor(indices=y_index, values=y_labels, shape=[batch_size, max_input_timesteps]), x_length))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

decoder = ctc.ctc_beam_search_decoder(prediction, x_length)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    prev_output_time = datetime.datetime.now()
    # Keep training until reach max iterations

    test_xs = word_dataset.get_test_data(max_input_timesteps)
    test_xs_length = word_dataset.get_test_sequence_lengths(max_input_timesteps)

    test_ys_index, test_ys_labels = word_dataset.get_test_labels_with_timesteps(max_input_timesteps)

    print("Test index: ", test_ys_index)
    print("Test labels: ", test_ys_labels)

    print("Start training")
    while step < learning_iterations:

        word_dataset.prepare_next_train_batch(batch_size)

        batch_xs = word_dataset.get_train_batch_data(max_input_timesteps)
        batch_xs_length = word_dataset.get_train_batch_sequence_lengths(max_input_timesteps)

        batch_ys_index, batch_ys_labels = word_dataset.get_train_batch_labels_with_timesteps(max_input_timesteps)

        sess.run(optimizer, feed_dict={x: batch_xs,
                                       x_length: batch_xs_length,
                                       y_index: batch_ys_index,
                                       y_labels: batch_ys_labels
                                       })

        from_prev_output_time = datetime.datetime.now() - prev_output_time

        if step == 1 or from_prev_output_time.seconds > display_time:
            batch_loss = sess.run(cost, feed_dict={x: batch_xs,
                                                   x_length: batch_xs_length,
                                                   y_index: batch_ys_index,
                                                   y_labels: batch_ys_labels
                                                   })

            test_decoded, _ = sess.run(decoder, feed_dict={x: test_xs,
                                                           x_length: test_xs_length
                                                           })

            print("=====")
            print("Step %d batch loss %f " % (step, batch_loss))
            print("-----")
            print("Test index decoded: ", test_decoded.indices)
            print("Test labels decoded: ", test_decoded.values)
            print("=====")

            prev_output_time = datetime.datetime.now()
        step += 1
