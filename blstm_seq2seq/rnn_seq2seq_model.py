import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import random
import os
import prepare_features as pf

def define_seq2seq_rnn_for_prediction(image_rnn_input_data,image_rnn_input_lengths,label_rnn_input_data):
    default_dropout_prob = tf.constant(1, "float")
    dropout_input_keep_prob = tf.placeholder_with_default(default_dropout_prob, default_dropout_prob.get_shape())
    dropout_output_keep_prob = tf.placeholder_with_default(default_dropout_prob, default_dropout_prob.get_shape())
    return define_seq2seq_rnn_for_training(image_rnn_input_data,image_rnn_input_lengths,label_rnn_input_data,dropout_input_keep_prob,dropout_output_keep_prob)

def define_seq2seq_rnn_for_training(image_rnn_input_data,image_rnn_input_lengths,label_rnn_input_data,dropout_input_keep_prob,dropout_output_keep_prob):
    # image_rnn_input_data (n_batch_size, n_steps, n_features)
    # label_rnn_input_data (n_batch_size, n_label_rnn_steps, n_classes)

    # Define RNN architecture
    n_image_rnn_cells = 1
    n_image_rnn_hidden = 64  # hidden layer num of features
    print("Image LSTM cells:", n_image_rnn_cells, "Image LSTM hidden units:", n_image_rnn_hidden)
    n_label_rnn_cells = 1
    n_label_rnn_hidden = 64  # hidden layer num of features
    print("Label LSTM cells:", n_label_rnn_cells, "Label LSTM hidden units:", n_label_rnn_hidden)

    # Retrieve dimensions from input data
    image_batch_size = tf.shape(image_rnn_input_data)[0]
    n_image_rnn_steps = image_rnn_input_data.get_shape()[1].value  # Timesteps = image width
    n_image_features = image_rnn_input_data.get_shape()[2].value

    label_batch_size = tf.shape(label_rnn_input_data)[0]
    n_label_rnn_steps = label_rnn_input_data.get_shape()[1].value
    n_classes = label_rnn_input_data.get_shape()[2].value

    # print(n_image_rnn_steps,n_image_features)
    # print(n_label_rnn_steps,n_classes)

    # Define weights
    w_image_hidden = tf.Variable(tf.random_normal([n_image_features, n_image_rnn_hidden]))
    b_image_hidden = tf.Variable(tf.random_normal([n_image_rnn_hidden]))
    w_label_hidden = tf.Variable(tf.random_normal([n_classes, n_label_rnn_hidden]))
    b_label_hidden = tf.Variable(tf.random_normal([n_label_rnn_hidden]))
    w_label_out = tf.Variable(tf.random_normal([n_label_rnn_hidden, n_classes]))
    b_label_out = tf.Variable(tf.random_normal([n_classes]))

    # Transform input data for image RNN
    image_rnn_inputs = tf.transpose(image_rnn_input_data, [1, 0, 2])  # (n_input_steps,n_batch_size,n_features)
    image_rnn_inputs = tf.reshape(image_rnn_inputs, [-1,
                                                     n_image_features])  # (n_steps*n_batch_size, n_features) (2D list with 28*256 vectors with 28 features each)
    image_rnn_inputs = tf.matmul(image_rnn_inputs,
                                 w_image_hidden) + b_image_hidden  # (n_steps*n_batch_size=28*256,n_hidden=128)
    image_rnn_inputs = tf.split(0, n_image_rnn_steps,
                                image_rnn_inputs)  # [(n_batch_size, n_features),(n_batch_size, n_features),...,(n_batch_size, n_features)]

    # Transform input data for label RNN
    label_rnn_inputs = tf.transpose(label_rnn_input_data, [1, 0, 2])  # (n_output_steps,n_batch_size,n_classes)
    label_rnn_inputs = tf.reshape(label_rnn_inputs, [-1,
                                                     n_classes])  # (n_steps*n_batch_size, n_features) (2D list with 28*256 vectors with 28 features each)
    label_rnn_inputs = tf.matmul(label_rnn_inputs,
                                 w_label_hidden) + b_label_hidden  # (n_steps*n_batch_size=28*256,n_hidden=128)
    label_rnn_inputs = tf.split(0, n_label_rnn_steps,
                                label_rnn_inputs)  # [(n_batch_size, n_features),(n_batch_size, n_features),...,(n_batch_size, n_features)]

    # Transform target data for label RNN
    # label_rnn_target_outputs = tf.transpose(label_rnn_target_data, [1, 0])  # (n_label_rnn_steps,n_batch_size)
    # label_rnn_target_outputs = tf.split(0, n_label_rnn_steps, label_rnn_target_outputs)
    # label_rnn_target_outputs = [tf.squeeze(lrt) for lrt in label_rnn_target_outputs]

    # Image RNN
    image_lstm_cell = rnn_cell.LSTMCell(n_image_rnn_hidden)
    image_lstm_cell = rnn_cell.DropoutWrapper(image_lstm_cell, input_keep_prob=dropout_input_keep_prob, output_keep_prob=dropout_output_keep_prob)
    if n_image_rnn_cells > 1:
        image_lstm_cell = rnn_cell.MultiRNNCell([image_lstm_cell] * n_image_rnn_cells)
    image_rnn_initial_state = image_lstm_cell.zero_state(image_batch_size, tf.float32)
    image_rnn_outputs, image_rnn_states = rnn.rnn(image_lstm_cell, image_rnn_inputs, initial_state=image_rnn_initial_state, sequence_length=image_rnn_input_lengths, scope="RNN1")
    # image_lstm_fw_cell = rnn_cell.LSTMCell(n_image_rnn_hidden, forget_bias=0)
    # image_lstm_fw_cell = rnn_cell.DropoutWrapper(image_lstm_fw_cell, input_keep_prob=dropout_input_keep_prob,
    #                                              output_keep_prob=dropout_output_keep_prob)
    # if n_image_rnn_cells > 1:
    #     image_lstm_fw_cell = rnn_cell.MultiRNNCell([image_lstm_fw_cell] * n_image_rnn_cells)
    # image_rnn_initial_state_fw = image_lstm_fw_cell.zero_state(image_batch_size, tf.float32)
    #
    # image_lstm_bw_cell = rnn_cell.LSTMCell(n_image_rnn_hidden, forget_bias=0)
    # image_lstm_bw_cell = rnn_cell.DropoutWrapper(image_lstm_bw_cell, input_keep_prob=dropout_input_keep_prob,
    #                                              output_keep_prob=dropout_output_keep_prob)
    # if n_image_rnn_cells > 1:
    #     image_lstm_bw_cell = rnn_cell.MultiRNNCell([image_lstm_bw_cell] * n_image_rnn_cells)
    # image_rnn_initial_state_bw = image_lstm_bw_cell.zero_state(image_batch_size, tf.float32)
    #
    # image_rnn_outputs, image_rnn_state_fw, image_rnn_state_bw = rnn.bidirectional_rnn(image_lstm_fw_cell,
    #                                                                                   image_lstm_bw_cell,
    #                                                                                   image_rnn_inputs,
    #                                                                                   initial_state_fw=image_rnn_initial_state_fw,
    #                                                                                   initial_state_bw=image_rnn_initial_state_bw)

    image_rnn_output = image_rnn_outputs[-1]

    # Label RNN
    label_lstm_cell = rnn_cell.LSTMCell(n_label_rnn_hidden, forget_bias=0)
    label_lstm_cell = rnn_cell.DropoutWrapper(label_lstm_cell, input_keep_prob=dropout_input_keep_prob,
                                              output_keep_prob=dropout_output_keep_prob)
    if n_label_rnn_cells > 1:
        label_lstm_cell = rnn_cell.MultiRNNCell([label_lstm_cell] * n_label_rnn_cells)

    label_rnn_initial_state = image_rnn_output
    label_rnn_initial_state = label_lstm_cell.zero_state(label_batch_size, tf.float32)
    w_image2label = tf.Variable(
        tf.random_normal([image_rnn_output.get_shape()[1].value, label_rnn_initial_state.get_shape()[1].value]))
    b_image2label = tf.Variable(tf.random_normal([label_rnn_initial_state.get_shape()[1].value]))
    label_rnn_initial_state = tf.tanh(tf.matmul(image_rnn_output, w_image2label) + b_image2label)

    label_rnn_outputs, label_rnn_states = rnn.rnn(label_lstm_cell, label_rnn_inputs,
                                                  initial_state=label_rnn_initial_state, scope="RNN2")

    label_rnn_outputs = [tf.matmul(lro, w_label_out) + b_label_out for lro in
                         label_rnn_outputs]  # n_label_rnn_steps * (n_batch_size,n_classes)

    label_rnn_predicted_index_labels = tf.pack(label_rnn_outputs)  # (n_label_rnn_steps,n_batch_size,n_classes)
    label_rnn_predicted_index_labels = tf.transpose(label_rnn_predicted_index_labels,
                                                    [1, 0, 2])  # (n_batch_size,n_label_rnn_steps,n_classes)
    label_rnn_predicted_index_labels = tf.argmax(label_rnn_predicted_index_labels,
                                                 2)  # (n_batch_size, n_label_rnn_steps)

    return label_rnn_outputs,label_rnn_predicted_index_labels


# Iterate label RNN to get final result
def get_label_rnn_result(label_rnn_predicted_index_labels,image_rnn_input_data,image_rnn_input_lengths,label_rnn_input_data,unique_chars,image_data):
    sess = tf.get_default_session()
    n_label_rnn_steps = label_rnn_input_data.get_shape()[1].value
    image_lengths = [len(f) for f in image_data]
    predicted_text_labels = [""] * len(image_data)
    for i in range(n_label_rnn_steps):
        input_labels = predicted_text_labels
        input_labels = [pf.START_WORD_CHAR + label for label in input_labels]  # Add start-word character
        input_labels = [label[:n_label_rnn_steps] for label in input_labels]  # Set fixed size length
        input_labels = [label.ljust(n_label_rnn_steps) for label in input_labels]  # Pad with spaces to right size
        one_hot_input_labels = pf.get_one_hot_labels(unique_chars,input_labels)  # Transform to one-hot labels

        predicted_index_labels = sess.run(label_rnn_predicted_index_labels,
                                          feed_dict={image_rnn_input_data: image_data,
                                                     image_rnn_input_lengths: image_lengths,
                                                     label_rnn_input_data: one_hot_input_labels})
        predicted_text_labels = pf.get_text_labels(unique_chars,predicted_index_labels)
        predicted_text_labels = [label[:i + 1] for label in
                                 predicted_text_labels]  # Cut off anything after current step
    return predicted_text_labels
