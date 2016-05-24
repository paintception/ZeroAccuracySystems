import datetime
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import word_dataset as wd
from word_dataset import WordDataSet,WordDataItem
import dirs
import random
import os
from tensorflow.python.ops.seq2seq import sequence_loss
import metrics

# Read data set
fixed_timestep_count = 50
max_image_width = 50
dataset = WordDataSet(dirs.KNMP_PROCESSED_WORD_BOXES_DIR_PATH,max_image_width=max_image_width)

print("Total items:",dataset.get_total_item_count())
print("Training items:",dataset.get_train_item_count())
n_test_items = dataset.get_test_item_count()
print("Test items:",n_test_items)
print("Max time steps (width):",dataset.get_max_time_steps())
n_label_rnn_steps = dataset.get_max_label_length() + 1
print("Max label length:", n_label_rnn_steps)

# Parameters
learning_rate = 0.001
print("Learning rate:",learning_rate)
n_batch_size = 128
print("Batch size:",n_batch_size)
dropout_input_keep_prob_value = 0.5
print('Dropout input keep probability:',dropout_input_keep_prob_value)
dropout_output_keep_prob_value = 0.5
print('Dropout output keep probability:',dropout_output_keep_prob_value)
n_classes = len(dataset.get_unique_chars()) # Classes (A,a,B,b,c,...)
print("Classes:",n_classes)
n_image_features = dataset.get_feature_count() # Features = image height
print("Features:", n_image_features)
n_image_rnn_steps = fixed_timestep_count # Timesteps = image width
print("Time steps:", n_image_rnn_steps)
n_image_rnn_cells = 1
n_image_rnn_hidden = 128 # hidden layer num of features
print("Image LSTM cells:", n_image_rnn_cells, "Image LSTM hidden units:", n_image_rnn_hidden)
n_label_rnn_cells = 1
n_label_rnn_hidden = 64 # hidden layer num of features
print("Label LSTM cells:", n_label_rnn_cells, "Label LSTM hidden units:", n_label_rnn_hidden)
display_time_interval_sec = 10

# Saved models
model_dir_path = dirs.KNMP_MODEL_DIR_PATH
last_model_file_path = os.path.join(model_dir_path,"last.model")
max_acc_model_file_path = os.path.join(model_dir_path,"max_acc.model")
if not os.path.exists(model_dir_path):
    os.makedirs(model_dir_path)

# Placeholders
default_dropout_prob = tf.constant(1,"float")
dropout_input_keep_prob = tf.placeholder_with_default(default_dropout_prob,[])
dropout_output_keep_prob = tf.placeholder_with_default(default_dropout_prob,[])
image_rnn_input_data = tf.placeholder("float", [None, n_image_rnn_steps, n_image_features]) # (n_batch_size, n_steps, n_features)
image_batch_size = tf.shape(image_rnn_input_data)[0]
label_rnn_input_data = tf.placeholder("float", [None, n_label_rnn_steps, n_classes])
label_batch_size = tf.shape(label_rnn_input_data)[0]
label_rnn_target_data = tf.placeholder("int64", [None, n_label_rnn_steps]) # (n_batch_size,n_label_rnn_steps)

# Weights
w_image_hidden = tf.Variable(tf.random_normal([n_image_features, n_image_rnn_hidden]))
w_image_hidden_sum = tf.reduce_sum(w_image_hidden)
b_image_hidden = tf.Variable(tf.random_normal([n_image_rnn_hidden]))
w_label_hidden = tf.Variable(tf.random_normal([n_classes, n_label_rnn_hidden]))
b_label_hidden = tf.Variable(tf.random_normal([n_label_rnn_hidden]))
w_label_out = tf.Variable(tf.random_normal([n_label_rnn_hidden,n_classes]))
b_label_out = tf.Variable(tf.random_normal([n_classes]))

# Transform input data for image RNN
image_rnn_inputs = tf.transpose(image_rnn_input_data, [1, 0, 2]) # (n_input_steps,n_batch_size,n_features)
image_rnn_inputs = tf.reshape(image_rnn_inputs, [-1, n_image_features]) # (n_steps*n_batch_size, n_features) (2D list with 28*256 vectors with 28 features each)
image_rnn_inputs = tf.matmul(image_rnn_inputs, w_image_hidden) + b_image_hidden  # (n_steps*n_batch_size=28*256,n_hidden=128)
image_rnn_inputs = tf.split(0, n_image_rnn_steps, image_rnn_inputs)  # [(n_batch_size, n_features),(n_batch_size, n_features),...,(n_batch_size, n_features)]

# Transform input data for label RNN
label_rnn_inputs = tf.transpose(label_rnn_input_data, [1, 0, 2]) # (n_output_steps,n_batch_size,n_classes)
label_rnn_inputs = tf.reshape(label_rnn_inputs, [-1, n_classes]) # (n_steps*n_batch_size, n_features) (2D list with 28*256 vectors with 28 features each)
label_rnn_inputs = tf.matmul(label_rnn_inputs, w_label_hidden) + b_label_hidden  # (n_steps*n_batch_size=28*256,n_hidden=128)
label_rnn_inputs = tf.split(0, n_label_rnn_steps, label_rnn_inputs)  # [(n_batch_size, n_features),(n_batch_size, n_features),...,(n_batch_size, n_features)]

# Transform target data for label RNN
label_rnn_target_outputs = tf.transpose(label_rnn_target_data, [1, 0]) # (n_label_rnn_steps,n_batch_size)
label_rnn_target_outputs = tf.split(0,n_label_rnn_steps,label_rnn_target_outputs)
label_rnn_target_outputs = [tf.squeeze(lrt) for lrt in label_rnn_target_outputs]

# Image RNN
# image_lstm_cell = rnn_cell.LSTMCell(n_image_rnn_hidden)
# image_lstm_cell = rnn_cell.DropoutWrapper(image_lstm_cell, input_keep_prob=dropout_input_keep_prob, output_keep_prob=dropout_output_keep_prob)
# if n_image_rnn_cells > 1:
#     image_lstm_cell = rnn_cell.MultiRNNCell([image_lstm_cell] * n_image_rnn_cells)
# image_rnn_initial_state = image_lstm_cell.zero_state(image_batch_size, tf.float32)
# image_rnn_outputs, image_rnn_states = rnn.rnn(image_lstm_cell, image_rnn_inputs, initial_state=image_rnn_initial_state, scope="RNN1")
image_lstm_fw_cell = rnn_cell.LSTMCell(n_image_rnn_hidden)
image_lstm_fw_cell = rnn_cell.DropoutWrapper(image_lstm_fw_cell, input_keep_prob=dropout_input_keep_prob, output_keep_prob=dropout_output_keep_prob)
if n_image_rnn_cells > 1:
    image_lstm_fw_cell = rnn_cell.MultiRNNCell([image_lstm_fw_cell] * n_image_rnn_cells)
image_rnn_initial_state_fw = image_lstm_fw_cell.zero_state(image_batch_size, tf.float32)

image_lstm_bw_cell = rnn_cell.LSTMCell(n_image_rnn_hidden)
image_lstm_bw_cell = rnn_cell.DropoutWrapper(image_lstm_bw_cell, input_keep_prob=dropout_input_keep_prob, output_keep_prob=dropout_output_keep_prob)
if n_image_rnn_cells > 1:
    image_lstm_bw_cell = rnn_cell.MultiRNNCell([image_lstm_bw_cell] * n_image_rnn_cells)
image_rnn_initial_state_bw = image_lstm_bw_cell.zero_state(image_batch_size, tf.float32)

image_rnn_outputs, image_rnn_state_fw, image_rnn_state_bw = rnn.bidirectional_rnn(image_lstm_fw_cell, image_lstm_bw_cell,
                                                                              image_rnn_inputs,
                                                                              initial_state_fw=image_rnn_initial_state_fw,
                                                                              initial_state_bw=image_rnn_initial_state_bw)

image_rnn_output = image_rnn_outputs[-1]

# Label RNN
label_lstm_cell = rnn_cell.LSTMCell(n_label_rnn_hidden)
label_lstm_cell = rnn_cell.DropoutWrapper(label_lstm_cell, input_keep_prob=dropout_input_keep_prob, output_keep_prob=dropout_output_keep_prob)
if n_label_rnn_cells > 1:
    label_lstm_cell = rnn_cell.MultiRNNCell([label_lstm_cell] * n_label_rnn_cells)

# label_rnn_initial_state = image_rnn_output
# label_rnn_initial_state = label_lstm_cell.zero_state(label_batch_size, tf.float32)
# w_image2label = tf.Variable(tf.random_normal([image_rnn_output.get_shape()[1].value, label_rnn_initial_state.get_shape()[1].value]))
# b_image2label = tf.Variable(tf.random_normal([label_rnn_initial_state.get_shape()[1].value]))
# label_rnn_initial_state = tf.tanh(tf.matmul(image_rnn_output, w_image2label) + b_image2label)

label_rnn_initial_state = image_rnn_output

label_rnn_outputs, label_rnn_states = rnn.rnn(label_lstm_cell, label_rnn_inputs, initial_state=label_rnn_initial_state, scope="RNN2")

label_rnn_outputs = [tf.matmul(lro, w_label_out) + b_label_out for lro in label_rnn_outputs] # n_label_rnn_steps * (n_batch_size,n_classes)

label_rnn_predicted_index_labels = tf.pack(label_rnn_outputs) # (n_label_rnn_steps,n_batch_size,n_classes)
label_rnn_predicted_index_labels = tf.transpose(label_rnn_predicted_index_labels,[1,0,2]) # (n_batch_size,n_label_rnn_steps,n_classes)
label_rnn_predicted_index_labels = tf.argmax(label_rnn_predicted_index_labels,2) # (n_batch_size, n_label_rnn_steps)

# Optimization

weights_shape = tf.shape(label_rnn_target_outputs[0])
sequence_loss_weights = [tf.ones(weights_shape)]*n_label_rnn_steps
sequence_loss_weight_value = 10.0
# Higher weights for first characters 10,5,2.5,1.25,...
for i in range(len(sequence_loss_weights)):
    sequence_loss_weights[i] = tf.fill(weights_shape,sequence_loss_weight_value)
    sequence_loss_weight_value = sequence_loss_weight_value * 0.7
cost = sequence_loss(label_rnn_outputs,label_rnn_target_outputs,sequence_loss_weights)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Initializing the variables
init = tf.initialize_all_variables()

# EXECUTION

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Restore model, if necessary
    restore_saver = tf.train.Saver()
    restore_saver.restore(sess, last_model_file_path)

    step = 1
    prev_output_time = datetime.datetime.now()
    best_accuracy = 0
    train_sample_losses = []

    # Training sample (every time the same)
    random.seed(0)
    dataset.prepare_next_train_batch(n_test_items)
    train_sample_data = dataset.get_train_batch_data(time_step_count=fixed_timestep_count)  # (batch_size,n_steps,n_input)
    train_sample_one_hot_labels = dataset.get_train_batch_fixed_length_one_hot_labels(n_label_rnn_steps,
                                                                               start_word_char=True)  # (batch_size,n_output_steps,n_classes)
    train_sample_index_labels = dataset.get_train_batch_fixed_length_index_labels(n_label_rnn_steps)  # (batch_size,n_output_steps)
    train_sample_text_labels = dataset.get_text_labels(train_sample_index_labels) # (batch_size)
    random.seed()

    # Test data
    test_data = dataset.get_test_data(time_step_count=fixed_timestep_count)  # (batch_size,n_steps,n_input)
    test_index_labels = dataset.get_test_fixed_length_index_labels(
        n_label_rnn_steps)  # (batch_size,n_output_steps)
    test_text_labels = dataset.get_text_labels(test_index_labels) # (batch_size)

    # Iterate label RNN to get final result
    def get_label_rnn_result(image_data):
        predicted_text_labels = [""] * len(image_data)
        for i in range(n_label_rnn_steps):
            input_labels = predicted_text_labels
            input_labels = [wd.START_WORD_CHAR + label for label in input_labels]  # Add start-word character
            input_labels = [label[:n_label_rnn_steps] for label in input_labels]  # Set fixed size length
            input_labels = [label.ljust(n_label_rnn_steps) for label in input_labels]  # Pad with spaces to right size
            one_hot_input_labels = dataset.get_one_hot_labels(input_labels)  # Transform to one-hot labels

            predicted_index_labels = sess.run(label_rnn_predicted_index_labels,
                                              feed_dict={image_rnn_input_data: image_data,
                                                         label_rnn_input_data: one_hot_input_labels})
            predicted_text_labels = dataset.get_text_labels(predicted_index_labels)
            predicted_text_labels = [label[:i + 1] for label in
                                     predicted_text_labels]  # Cut off anything after current step
        return predicted_text_labels

    while True:
        # Training
        dataset.prepare_next_train_batch(n_batch_size)
        train_batch_data = dataset.get_train_batch_data(time_step_count=fixed_timestep_count)  # (batch_size,n_steps,n_input)
        train_batch_one_hot_labels = dataset.get_train_batch_fixed_length_one_hot_labels(n_label_rnn_steps, start_word_char=True) # (batch_size,n_output_steps,n_classes)
        train_batch_index_labels = dataset.get_train_batch_fixed_length_index_labels(n_label_rnn_steps) # (batch_size,n_output_steps)

        sess.run(optimizer, feed_dict={image_rnn_input_data:train_batch_data,
                                       label_rnn_input_data:train_batch_one_hot_labels,
                                       label_rnn_target_data:train_batch_index_labels,
                                       dropout_input_keep_prob:dropout_input_keep_prob_value,
                                       dropout_output_keep_prob:dropout_output_keep_prob_value})

        from_prev_output_time = datetime.datetime.now() - prev_output_time
        if step == 1 or from_prev_output_time.seconds > display_time_interval_sec:
            train_sample_loss = sess.run(cost,
                                         feed_dict={image_rnn_input_data:train_sample_data,
                                label_rnn_input_data: train_sample_one_hot_labels,
                                label_rnn_target_data: train_sample_index_labels})
            train_sample_losses.append(train_sample_loss)
            avg_count = 10
            last_batch_losses = train_sample_losses[-min(avg_count, len(train_sample_losses)):]
            average_batch_loss = sum(last_batch_losses) / len(last_batch_losses)

            predicted_train_sample_text_labels = get_label_rnn_result(train_sample_data)
            train_sample_word_level_accuracy = metrics.get_word_level_accuracy(train_sample_text_labels, predicted_train_sample_text_labels)
            train_sample_char_level_accuracy = metrics.get_char_level_accuracy(train_sample_text_labels, predicted_train_sample_text_labels)
            train_sample_avg_word_distance = metrics.get_avg_word_distance(train_sample_text_labels, predicted_train_sample_text_labels)

            predicted_test_text_labels = get_label_rnn_result(test_data)
            test_word_level_accuracy = metrics.get_word_level_accuracy(test_text_labels,predicted_test_text_labels)
            test_char_level_accuracy = metrics.get_char_level_accuracy(test_text_labels,predicted_test_text_labels)
            test_avg_word_distance = metrics.get_avg_word_distance(test_text_labels, predicted_test_text_labels)

            print("Iter:", step * n_batch_size, "TRAINING Loss: " + "{:.5f}".format(train_sample_loss), "[{:.5f}]".format(average_batch_loss), "Word acc.: " + "{:.4f}".format(train_sample_word_level_accuracy), "Char acc.: " + "{:.4f}".format(train_sample_char_level_accuracy), "Levenstein: " + "{:.4f}".format(train_sample_avg_word_distance),
                  "TEST","Word acc.: " + "{:.4f}".format(test_word_level_accuracy), "Char acc.: " + "{:.4f}".format(test_char_level_accuracy), "Levenstein: " + "{:.4f}".format(test_avg_word_distance),
                  "*" if test_avg_word_distance > best_accuracy else "")
            print("TRAINING SAMPLE:")
            print(train_sample_text_labels)
            print(predicted_train_sample_text_labels)
            print("TEST:")
            print(test_text_labels)
            print(predicted_test_text_labels)
            #print(sess.run(w_image_hidden_sum))
            print()

            saver = tf.train.Saver()
            saver.save(sess, last_model_file_path)

            if test_avg_word_distance > best_accuracy:
                best_accuracy = test_avg_word_distance
                saver.save(sess, max_acc_model_file_path)

            prev_output_time = datetime.datetime.now()

        step += 1

    sess.close()
