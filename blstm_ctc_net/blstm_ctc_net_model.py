import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
from tensorflow.contrib import ctc

import os
import sys
import datetime

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import blstm_ctc_net.word_dataset_with_timesteps as wd
import blstm_ctc_net.dirs as dirs
import metrics
#import blstm_ctc_net.plot_words as plotter
from word_model.word_m import WordM, WordMKNMP


# Override tensorflow function with a bug
from tensorflow.python.ops import gen_ctc_ops
from tensorflow.python.framework import ops


def ctc_beam_search(inputs, sequence_length, beam_width=100,
                            top_paths=1, merge_repeated=True):
    decoded_ixs, decoded_vals, decoded_shapes, log_probabilities = (
        gen_ctc_ops._ctc_beam_search_decoder(
            inputs, sequence_length, beam_width=beam_width, top_paths=top_paths,
            merge_repeated=merge_repeated))
    return ops.SparseTensor(decoded_ixs[0], decoded_vals[0], decoded_shapes[0]), log_probabilities

ctc.ctc_beam_search_decoder = ctc_beam_search
# =========================================

print("Loading data")
if __name__ == "__main__":
    # word_dataset = wd.WordDataSet(dir_path=[dirs.KNMP_PROCESSED_WORD_BOXES_DIR_PATH, dirs.STANFORD_PROCESSED_WORD_BOXES_DIR_PATH])
    word_dataset = wd.WordDataSet(dir_path=[dirs.KNMP_PROCESSED_WORD_BOXES_DIR_PATH])
else:
    word_dataset = wd.WordDataSet(dir_path=[dirs.KNMP_PROCESSED_WORD_BOXES_DIR_PATH], train=False)
    # word_dataset = wd.WordDataSet(dir_path=dirs.STANFORD_PROCESSED_WORD_BOXES_DIR_PATH, train=False)

word_model = WordMKNMP()

save_path = os.path.join(os.path.dirname(__file__), "last_model", "current.model")

display_time = 60

# Global parameters
learning_rate = 0.0001
learning_iterations = 100000  # number of mini-batches
batch_size = 128  # number of words in a batch
n_hidden_layer = 128  # number of nodes in hidden layer
n_output_classes = len(word_dataset.unique_chars) + 1  # Number of letters in our alphabet and empty label
print("Output_classes: ", n_output_classes)

n_input = 28  # Number of input features for each sliding window of 1px
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
def blstm_layer(_X, _x_length, batch_s):
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

    istate_fw = lstm_fw_cell.zero_state(batch_s, tf.float32)
    istate_bw = lstm_bw_cell.zero_state(batch_s, tf.float32)

    # Get lstm cell output
    outputs, output_state_fw, output_state_bw = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, _X,
                                                                      initial_state_fw=istate_fw,
                                                                      initial_state_bw=istate_bw,
                                                                      sequence_length=_x_length
                                                                      )

    outputs = tf.concat(0, outputs)
    activation = tf.matmul(outputs, output_weights) + output_biases

    return tf.reshape(activation, [max_input_timesteps, batch_s, n_output_classes])


def blstm_ctc_train():
    prediction = blstm_layer(x, x_length, batch_size)
    target_labels = tf.SparseTensor(indices=y_index, values=y_labels, shape=[batch_size, max_input_timesteps])
    cost = tf.reduce_mean(ctc.ctc_loss(prediction, target_labels, x_length))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    decoder = ctc.ctc_beam_search_decoder(prediction, x_length)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        saver = tf.train.Saver()
        if os.path.exists(save_path):
            saver.restore(sess, save_path)

        step = 1
        prev_output_time = datetime.datetime.now()

        # test_xs_e = word_dataset.get_test_data(0)

        test_xs = word_dataset.get_test_data(max_input_timesteps)
        test_xs_length = word_dataset.get_test_sequence_lengths(max_input_timesteps)

        test_ys_index, test_ys_labels = word_dataset.get_test_labels_with_timesteps(max_input_timesteps)

        # print(test_ys_index)
        # print(test_ys_labels)

        test_words = word_dataset.get_words_from_indexes(test_ys_index, word_dataset.get_chars_from_indexes(test_ys_labels), batch_size)
        print("Start training")
        while step < learning_iterations:

            word_dataset.prepare_next_train_batch(batch_size)

            batch_xs = word_dataset.get_train_batch_data(max_input_timesteps)
            batch_xs_length = word_dataset.get_train_batch_sequence_lengths(max_input_timesteps)

            batch_ys_index, batch_ys_labels = word_dataset.get_train_batch_labels_with_timesteps(max_input_timesteps)

            # print(batch_ys_index)
            # print(batch_ys_labels)

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

                batch_decoded, _ = sess.run(decoder, feed_dict={x: batch_xs,
                                                                x_length: batch_xs_length,
                                                                })

                test_decoded, _ = sess.run(decoder, feed_dict={x: test_xs,
                                                               x_length: test_xs_length
                                                               })

                test_words_decoded = word_dataset.get_words_from_indexes(test_decoded.indices,
                                                                         word_dataset.get_chars_from_indexes(test_decoded.values),
                                                                         batch_size)

                batch_words = word_dataset.get_words_from_indexes(batch_ys_index,
                                                                  word_dataset.get_chars_from_indexes(batch_ys_labels),
                                                                  batch_size)

                batch_words_decoded = word_dataset.get_words_from_indexes(batch_decoded.indices,
                                                                          word_dataset.get_chars_from_indexes(batch_decoded.values),
                                                                          batch_size)

                batch_words_decoded_lexicon = [word_model.get_closest_word(word) for word in batch_words_decoded]
                test_words_decoded_lexicon = [word_model.get_closest_word(word) for word in test_words_decoded]

                print("=====")
                print("Step %d" % step)
                print("Batch loss: ", batch_loss)
                print("-----")
                print("Batch accuracy words: %f chars: %f average distance: %f" % (metrics.get_word_level_accuracy(batch_words, batch_words_decoded),
                                                                                   metrics.get_char_level_accuracy(batch_words, batch_words_decoded),
                                                                                   metrics.get_avg_word_distance(batch_words, batch_words_decoded)))
                print("Test accuracy words: %f chars: %f average distance: %f" % (metrics.get_word_level_accuracy(test_words, test_words_decoded),
                                                                                  metrics.get_char_level_accuracy(test_words, test_words_decoded),
                                                                                  metrics.get_avg_word_distance(test_words, test_words_decoded)))
                print("-----")
                print("Batch lexicon accuracy words: %f chars: %f average distance: %f" % (metrics.get_word_level_accuracy(batch_words, batch_words_decoded_lexicon),
                                                                                           metrics.get_char_level_accuracy(batch_words, batch_words_decoded_lexicon),
                                                                                           metrics.get_avg_word_distance(batch_words, batch_words_decoded_lexicon)))
                print("Test lexicon accuracy words: %f chars: %f average distance: %f" % (metrics.get_word_level_accuracy(test_words, test_words_decoded_lexicon),
                                                                                          metrics.get_char_level_accuracy(test_words, test_words_decoded_lexicon),
                                                                                          metrics.get_avg_word_distance(test_words, test_words_decoded_lexicon)))
                print("-----")
                # print("Test labels")
                print([word.ljust(15) for word in test_words])
                print([word.ljust(15) for word in test_words_decoded])
                print([word.ljust(15) for word in test_words_decoded_lexicon])
                # print("Batch labels")
                # print([word.ljust(15) for word in batch_words])
                # print([word.ljust(15) for word in batch_words_decoded])
                # print([word.ljust(15) for word in batch_words_decoded_lexicon])

                # plotter.plot_words_with_labels(test_xs[0:17], test_words[0:17], test_words_decoded[0:17])

                saver.save(sess, save_path)

                prev_output_time = datetime.datetime.now()
            step += 1


def blstm_ctc_predict(batch_xs, batch_xs_length, batch_words):
    batch_s = len(batch_xs)

    prediction = blstm_layer(x, x_length, batch_s)

    decoder = ctc.ctc_beam_search_decoder(prediction, x_length)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        saver = tf.train.Saver()
        saver.restore(sess, save_path)

        batch_decoded, _ = sess.run(decoder, feed_dict={x: batch_xs,
                                                        x_length: batch_xs_length,
                                                        })

        batch_words_decoded = word_dataset.get_words_from_indexes(batch_decoded.indices,
                                                                  word_dataset.get_chars_from_indexes(batch_decoded.values),
                                                                  batch_s)

        batch_words_decoded_lexicon = [word_model.get_closest_word(word) for word in batch_words_decoded]

        # plotter.plot_words_with_labels(batch_xs, batch_words, batch_words_decoded)
        # plotter.plot_words_with_labels(batch_xs, batch_words, batch_words_decoded_lexicon)

        print("Batch accuracy words: %f chars: %f average distance: %f" % (metrics.get_word_level_accuracy(batch_words, batch_words_decoded),
                                                                           metrics.get_char_level_accuracy(batch_words, batch_words_decoded),
                                                                           metrics.get_avg_word_distance(batch_words, batch_words_decoded)))

        print("Batch lexicon accuracy words: %f chars: %f average distance: %f" % (metrics.get_word_level_accuracy(batch_words, batch_words_decoded_lexicon),
                                                                                   metrics.get_char_level_accuracy(batch_words, batch_words_decoded_lexicon),
                                                                                   metrics.get_avg_word_distance(batch_words, batch_words_decoded_lexicon)))

        print("Batch labels")
        print([word.ljust(15) for word in batch_words])
        print([word.ljust(15) for word in batch_words_decoded])
        print([word.ljust(15) for word in batch_words_decoded_lexicon])

        return batch_words_decoded_lexicon


if __name__ == "__main__":
    blstm_ctc_train()
