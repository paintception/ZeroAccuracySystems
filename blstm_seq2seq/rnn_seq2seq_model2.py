import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import random
import os
import prepare_features as pf

def define_seq2seq_rnn_for_prediction(image_input_data,image_input_lengths,label_rnn_input_data):
    default_dropout_prob = tf.constant(1, "float")
    dropout_input_keep_prob = tf.placeholder_with_default(default_dropout_prob, default_dropout_prob.get_shape())
    dropout_output_keep_prob = tf.placeholder_with_default(default_dropout_prob, default_dropout_prob.get_shape())
    return define_seq2seq_rnn_for_training(image_input_data,image_input_lengths,label_rnn_input_data,dropout_input_keep_prob,dropout_output_keep_prob)

def define_seq2seq_rnn_for_training(image_input_data,image_input_lengths,label_rnn_input_data,dropout_input_keep_prob,dropout_output_keep_prob):
    # image_rnn_input_data (n_batch_size, n_steps, n_features)
    # label_rnn_input_data (n_batch_size, n_label_rnn_steps, n_classes)

    # Convulation NN
    image_width = image_input_data.get_shape()[1].value
    image_height = image_input_data.get_shape()[2].value

    image_input_data_conv = tf.reshape(image_input_data, [-1, image_width, image_height, 1])

    n_conv1_patch_size = 7
    n_conv1_channels = 32
    print("Convolutional layer 1, Patch size:",n_conv1_patch_size,"Channels:",n_conv1_channels)
    w_conv1 = tf.Variable(tf.random_normal([n_conv1_patch_size, n_conv1_patch_size, 1, n_conv1_channels]),name="w_conv1")
    b_conv1 = tf.Variable(tf.random_normal([n_conv1_channels]),name="b_conv1")

    conv1 = tf.tanh(tf.nn.conv2d(image_input_data_conv, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

    # n_conv2_patch_size = 5
    # n_conv2_channels = 16
    # print("Convolutional layer 2, Patch size:", n_conv2_patch_size, "Channels:", n_conv2_channels)
    # w_conv2 = tf.Variable(tf.random_normal([n_conv2_patch_size, n_conv2_patch_size, n_conv1_channels, n_conv2_channels]),name="w_conv2")
    # b_conv2 = tf.Variable(tf.random_normal([n_conv2_channels]),name="b_conv2")
    #
    # conv2 = tf.tanh(tf.nn.conv2d(conv1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    #
    # n_conv3_patch_size = 5
    # n_conv3_channels = 16
    # print("Convolutional layer 3, Patch size:", n_conv3_patch_size, "Channels:", n_conv3_channels)
    # w_conv3 = tf.Variable(
    #     tf.random_normal([n_conv3_patch_size, n_conv3_patch_size, n_conv2_channels, n_conv3_channels]), name="w_conv3")
    # b_conv3 = tf.Variable(tf.random_normal([n_conv3_channels]), name="b_conv3")
    #
    # conv3 = tf.tanh(tf.nn.conv2d(conv2, w_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)

    image_rnn_inputs = tf.reshape(conv1, [-1, image_width, image_height*n_conv1_channels])

    # Define RNN architecture
    n_image_rnn_cells = 1
    n_image_rnn_hidden = 96  # hidden layer num of features
    print("Image LSTM cells:", n_image_rnn_cells, "Image LSTM hidden units:", n_image_rnn_hidden)
    n_label_rnn_cells = 1
    n_label_rnn_hidden = 96  # hidden layer num of features
    print("Label LSTM cells:", n_label_rnn_cells, "Label LSTM hidden units:", n_label_rnn_hidden)

    # Retrieve dimensions from input data
    image_batch_size = tf.shape(image_rnn_inputs)[0]
    n_image_rnn_steps = image_rnn_inputs.get_shape()[1].value  # Timesteps = image width
    n_image_features = image_rnn_inputs.get_shape()[2].value

    label_batch_size = tf.shape(label_rnn_input_data)[0]
    n_label_rnn_steps = label_rnn_input_data.get_shape()[1].value
    n_classes = label_rnn_input_data.get_shape()[2].value

    print(n_image_rnn_steps,n_image_features)
    print(n_label_rnn_steps,n_classes)

    # Define RNN weights
    w_label_hidden = tf.Variable(tf.random_normal([n_classes, n_label_rnn_hidden]),name="w_label_hidden")
    b_label_hidden = tf.Variable(tf.random_normal([n_label_rnn_hidden]),name="b_label_hidden")
    w_label_out = tf.Variable(tf.random_normal([n_label_rnn_hidden, n_classes]),name="w_label_out")
    b_label_out = tf.Variable(tf.random_normal([n_classes]),name="b_label_out")

    # Image RNN
    image_lstm_cell = rnn_cell.LSTMCell(n_image_rnn_hidden)
    image_lstm_cell = rnn_cell.DropoutWrapper(image_lstm_cell, input_keep_prob=dropout_input_keep_prob, output_keep_prob=dropout_output_keep_prob)
    if n_image_rnn_cells > 1:
        image_lstm_cell = rnn_cell.MultiRNNCell([image_lstm_cell] * n_image_rnn_cells)
    image_rnn_initial_state = image_lstm_cell.zero_state(image_batch_size, tf.float32)
    image_rnn_outputs, image_rnn_states = rnn.dynamic_rnn(image_lstm_cell, image_rnn_inputs, initial_state=image_rnn_initial_state, sequence_length=image_input_lengths, scope="RNN1")
    image_rnn_output = last_relevant(image_rnn_outputs,image_input_lengths)

    # Transform input data for label RNN
    label_rnn_inputs = tf.transpose(label_rnn_input_data, [1, 0, 2])  # (n_output_steps,n_batch_size,n_classes)
    label_rnn_inputs = tf.reshape(label_rnn_inputs, [-1,
                                                     n_classes])  # (n_steps*n_batch_size, n_features) (2D list with 28*256 vectors with 28 features each)
    label_rnn_inputs = tf.matmul(label_rnn_inputs,
                                 w_label_hidden) + b_label_hidden  # (n_steps*n_batch_size=28*256,n_hidden=128)
    label_rnn_inputs = tf.split(0, n_label_rnn_steps,
                                label_rnn_inputs)  # [(n_batch_size, n_features),(n_batch_size, n_features),...,(n_batch_size, n_features)]

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
def get_label_rnn_result(label_rnn_predicted_index_labels,image_rnn_input_data,image_rnn_input_lengths,label_rnn_input_data,unique_chars,image_data,image_lengths):
    sess = tf.get_default_session()
    n_label_rnn_steps = label_rnn_input_data.get_shape()[1].value
    #image_lengths = [len(f) for f in image_data]
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

def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = int(output.get_shape()[1])
    output_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, output_size])
    relevant = tf.gather(flat, index)
    return relevant