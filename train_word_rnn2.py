import datetime
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
from word_dataset import WordDataSet,WordDataItem
import dirs
import random
import os

# Read data set
fixed_timestep_count = 50
dataset = WordDataSet(dirs.KNMP_PROCESSED_WORD_BOXES_DIR_PATH)

print("Total items:",dataset.get_total_item_count())
print("Training items:",dataset.get_train_item_count())
print("Test items:",dataset.get_test_item_count())
print("Max label length:",dataset.get_max_label_length())

# Parameters
learning_rate = 0.0005
print("Learning rate:",learning_rate)
n_batch_size = 256
print("Batch size:",n_batch_size)
dropout_input_keep_prob_value = 0.5
print('Dropout input keep probability:',dropout_input_keep_prob_value)
dropout_output_keep_prob_value = 0.5
print('Dropout output keep probability:',dropout_output_keep_prob_value)
n_features = dataset.get_feature_count() # Features = image height
print("Features:",n_features)
n_steps = fixed_timestep_count # Timesteps = image width
print("Time steps:",n_steps)
n_cells = 2 # Number of cells/layers
print("Cells:", n_cells)
n_hidden = 128 # hidden layer num of features
print("Hidden units:",n_hidden)
n_classes = len(dataset.get_unique_chars()) # Classes (A,a,B,b,c,...)
print("Classes:",n_classes)
display_time_interval_sec = 60

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
x = tf.placeholder("float", [None, n_steps, n_features]) # (n_batch_size, n_steps, n_features)
y = tf.placeholder("float", [None, n_classes])
batch_size = tf.shape(x)[0]

# Weights
w_hidden = tf.Variable(tf.random_normal([n_features, n_hidden]))
b_hidden = tf.Variable(tf.random_normal([n_hidden]))
w_out = tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
b_out = tf.Variable(tf.random_normal([n_classes]))

# Transform input data for RNN (mystical part)
x1 = tf.transpose(x, [1, 0, 2]) # (n_steps,n_batch_size,n_features)
x2 = tf.reshape(x1, [-1, n_features]) # (n_steps*n_batch_size, n_features) (2D list with 28*256 vectors with 28 features each)
x_hidden = tf.matmul(x2, w_hidden) + b_hidden  # (n_steps*n_batch_size=28*256,n_hidden=128)
rnn_inputs = tf.split(0, n_steps, x_hidden)  # [(n_batch_size, n_features),(n_batch_size, n_features),...,(n_batch_size, n_features)]

# RNN
lstm_fw_cell = rnn_cell.LSTMCell(n_hidden)
lstm_fw_cell_dropout = rnn_cell.DropoutWrapper(lstm_fw_cell, input_keep_prob=dropout_input_keep_prob, output_keep_prob=dropout_output_keep_prob)
multi_lstm_fw_cell = lstm_fw_cell_dropout
if n_cells > 1:
    multi_lstm_fw_cell = rnn_cell.MultiRNNCell([lstm_fw_cell_dropout] * n_cells)

lstm_bw_cell = rnn_cell.LSTMCell(n_hidden)
lstm_bw_cell_dropout = rnn_cell.DropoutWrapper(lstm_bw_cell, input_keep_prob=dropout_input_keep_prob, output_keep_prob=dropout_output_keep_prob)
multi_lstm_bw_cell = lstm_bw_cell_dropout
if n_cells > 1:
    multi_lstm_bw_cell = rnn_cell.MultiRNNCell([lstm_bw_cell_dropout] * n_cells)

initial_state_fw = multi_lstm_fw_cell.zero_state(batch_size, tf.float32)
initial_state_bw = multi_lstm_bw_cell.zero_state(batch_size, tf.float32)

rnn_outputs, rnn_output_state_fw, rnn_output_state_bw = rnn.bidirectional_rnn(multi_lstm_fw_cell, multi_lstm_bw_cell, rnn_inputs,
                                initial_state_fw=initial_state_fw,
                                initial_state_bw=initial_state_bw)

rnn_output = rnn_outputs[-1]
y_pred = tf.matmul(rnn_output, w_out) + b_out

# Optimization
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

correct_pred = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# EXECUTION

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Restore model, if necessary
    restore_saver = tf.train.Saver()
    restore_saver.restore(sess, max_acc_model_file_path)

    step = 1
    prev_output_time = datetime.datetime.now()
    best_test_acc = 0
    batch_losses = []

    while True:
        # Training
        dataset.prepare_next_train_batch(n_batch_size)
        batch_xs = dataset.get_train_batch_data(time_step_count=fixed_timestep_count)  # (batch_size,n_steps,n_input)
        batch_ys = dataset.get_train_batch_first_char_one_hot_labels()  # (batch_size,n_classes)

        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                       dropout_input_keep_prob: dropout_input_keep_prob_value,
                                       dropout_output_keep_prob: dropout_output_keep_prob_value})

        from_prev_output_time = datetime.datetime.now() - prev_output_time
        if step == 1 or from_prev_output_time.seconds > display_time_interval_sec:
            # Calculate training batch accuracy
            batch_acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
            # Calculate training batch loss
            batch_loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})
            batch_losses.append(batch_loss)
            avg_count = 10
            last_batch_losses = batch_losses[-min(avg_count, len(batch_losses)):]
            average_batch_loss = sum(last_batch_losses) / len(last_batch_losses)

            # Calculate test accuracy
            test_xs = dataset.get_test_data(time_step_count=fixed_timestep_count)
            test_ys = dataset.get_test_first_char_one_hot_labels()

            test_acc = sess.run(accuracy, feed_dict={x: test_xs, y: test_ys})

            print ("Iteration " + str(step*n_batch_size) + ", Minibatch Loss = " + "{:.5f}".format(batch_loss) + \
                  " [{:.5f}]".format(average_batch_loss) + \
                  ", Training Accuracy = " + "{:.4f}".format(batch_acc) + \
                   ", Test Accuracy = " + "{:.4f}".format(test_acc),
                    "*" if test_acc > best_test_acc else "")

            saver = tf.train.Saver()
            saver.save(sess, last_model_file_path)

            if (test_acc > best_test_acc):
                best_test_acc = test_acc
            saver.save(sess, max_acc_model_file_path)

            prev_output_time = datetime.datetime.now()

        step += 1
