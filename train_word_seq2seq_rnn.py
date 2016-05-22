import datetime
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
from word_dataset import WordDataSet,WordDataItem
import dirs
import random
import os
from tensorflow.python.ops.seq2seq import sequence_loss

# Read data set
fixed_timestep_count = 50
max_image_width = 50
dataset = WordDataSet(dirs.KNMP_PROCESSED_WORD_BOXES_DIR_PATH,max_image_width=max_image_width)

print("Total items:",dataset.get_total_item_count())
print("Training items:",dataset.get_train_item_count())
print("Test items:",dataset.get_test_item_count())
print("Max time steps (width):",dataset.get_max_time_steps())
n_label_rnn_steps = dataset.get_max_label_length() + 1
print("Max label length:", n_label_rnn_steps)

# Parameters
learning_rate = 0.0005
print("Learning rate:",learning_rate)
n_batch_size = 256
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
n_image_rnn_hidden = 128 # hidden layer num of features
print("Image LSTM hidden units:", n_image_rnn_hidden)
n_label_rnn_hidden = 1000 # hidden layer num of features
print("Label LSTM hidden units:", n_label_rnn_hidden)
display_time_interval_sec = 5

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
label_rnn_input_data = tf.placeholder("float", [None, n_label_rnn_steps, n_classes])
label_rnn_target_data = tf.placeholder("int64", [None, n_label_rnn_steps]) # (n_batch_size,n_label_rnn_steps)
batch_size = tf.shape(image_rnn_input_data)[0]

# Weights
w_image_hidden = tf.Variable(tf.random_normal([n_image_features, n_image_rnn_hidden]))
b_image_hidden = tf.Variable(tf.random_normal([n_image_rnn_hidden]))
w_label_hidden = tf.Variable(tf.random_normal([n_classes, n_label_rnn_hidden]))
b_label_hidden = tf.Variable(tf.random_normal([n_label_rnn_hidden]))
w_image2label = tf.Variable(tf.random_normal([n_image_rnn_hidden, n_label_rnn_hidden]))
b_image2label = tf.Variable(tf.random_normal([n_label_rnn_hidden]))

# Transform input data for image RNN
image_rnn_input_data_trans = tf.transpose(image_rnn_input_data, [1, 0, 2]) # (n_input_steps,n_batch_size,n_features)
image_rnn_input_data_reshape = tf.reshape(image_rnn_input_data_trans, [-1, n_image_features]) # (n_steps*n_batch_size, n_features) (2D list with 28*256 vectors with 28 features each)
image_rnn_input_data_hidden = tf.matmul(image_rnn_input_data_reshape, w_image_hidden) + b_image_hidden  # (n_steps*n_batch_size=28*256,n_hidden=128)
image_rnn_inputs = tf.split(0, n_image_rnn_steps, image_rnn_input_data_hidden)  # [(n_batch_size, n_features),(n_batch_size, n_features),...,(n_batch_size, n_features)]

# Transform input data for label RNN
label_rnn_input_data_trans = tf.transpose(label_rnn_input_data, [1, 0, 2]) # (n_output_steps,n_batch_size,n_classes)
label_rnn_input_data_reshape = tf.reshape(label_rnn_input_data_trans, [-1, n_classes]) # (n_steps*n_batch_size, n_features) (2D list with 28*256 vectors with 28 features each)
label_rnn_input_data_hidden = tf.matmul(label_rnn_input_data_reshape, w_label_hidden) + b_label_hidden  # (n_steps*n_batch_size=28*256,n_hidden=128)
label_rnn_inputs = tf.split(0, n_label_rnn_steps, label_rnn_input_data_hidden)  # [(n_batch_size, n_features),(n_batch_size, n_features),...,(n_batch_size, n_features)]
# for output_rnn_input in output_rnn_inputs:
#     print(output_rnn_input)

# Image RNN
image_lstm_cell = rnn_cell.LSTMCell(n_image_rnn_hidden)
image_rnn_initial_state = image_lstm_cell.zero_state(batch_size, tf.float32)
image_rnn_outputs, image_rnn_states = rnn.rnn(image_lstm_cell, image_rnn_inputs, initial_state=image_rnn_initial_state, scope="RNN1")
image_rnn_output = image_rnn_outputs[-1]

# Label RNN
label_lstm_cell = rnn_cell.LSTMCell(n_label_rnn_hidden, num_proj=n_classes)
label_rnn_initial_state = label_lstm_cell.zero_state(batch_size, tf.float32)
#output_initial_state = tf.matmul(input_rnn_output, w_in_out) + b_in_out
label_rnn_outputs, label_rnn_states = rnn.rnn(label_lstm_cell, label_rnn_inputs, initial_state=label_rnn_initial_state, scope="RNN2")



label_rnn_target_data_trans = tf.transpose(label_rnn_target_data, [1, 0]) # (n_label_rnn_steps,n_batch_size)
label_rnn_target_split = tf.split(0,n_label_rnn_steps,label_rnn_target_data_trans)
label_rnn_target_outputs = [tf.squeeze(lrt) for lrt in label_rnn_target_split]

# for label_rnn_target_output in label_rnn_target_outputs:
#     print(label_rnn_target_output)

# label_rnn_outputs_conc = tf.concat(0,label_rnn_outputs)
# label_rnn_predicted_data = tf.reshape(label_rnn_outputs_conc,[n_label_rnn_steps,-1,n_classes])

# Optimization

#cost = tf.nn.sparse_softmax_cross_entropy_with_logits(label_rnn_predicted_data,label_rnn_target_data)
sequence_loss_weights = [tf.ones(tf.shape(label_rnn_target_outputs[0]))]*n_label_rnn_steps
cost = sequence_loss(label_rnn_outputs,label_rnn_target_outputs,sequence_loss_weights)

# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
#
# correct_pred = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# EXECUTION

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Restore model, if necessary
    # restore_saver = tf.train.Saver()
    # restore_saver.restore(sess, max_acc_model_file_path)

    step = 1
    prev_output_time = datetime.datetime.now()
    best_test_acc = 0
    batch_losses = []

    while True:
        # Training
        dataset.prepare_next_train_batch(n_batch_size)
        train_batch_data = dataset.get_train_batch_data(time_step_count=fixed_timestep_count)  # (batch_size,n_steps,n_input)
        train_one_hot_labels = dataset.get_train_batch_fixed_length_one_hot_labels(n_label_rnn_steps, start_word_char=True) # (batch_size,n_output_steps,n_classes)
        train_index_labels = dataset.get_train_batch_fixed_length_index_labels(n_label_rnn_steps) # (batch_size,n_output_steps)

        sess.run(optimizer, feed_dict={image_rnn_input_data: train_batch_data,label_rnn_input_data:train_one_hot_labels,label_rnn_target_data:train_index_labels})

        from_prev_output_time = datetime.datetime.now() - prev_output_time
        if step == 1 or from_prev_output_time.seconds > display_time_interval_sec:
            print(sess.run(cost,
                     feed_dict={image_rnn_input_data: train_batch_data, label_rnn_input_data: train_one_hot_labels,
                                label_rnn_target_data: train_index_labels}))

            # # Calculate training batch accuracy
            # batch_acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
            # # Calculate training batch loss
            # batch_loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})
            # batch_losses.append(batch_loss)
            # avg_count = 10
            # last_batch_losses = batch_losses[-min(avg_count, len(batch_losses)):]
            # average_batch_loss = sum(last_batch_losses) / len(last_batch_losses)
            #
            # # Calculate test accuracy
            # test_xs = dataset.get_test_data(time_step_count=fixed_timestep_count)
            # test_ys = dataset.get_test_first_char_one_hot_labels()
            #
            # test_acc = sess.run(accuracy, feed_dict={x: test_xs, y: test_ys})
            #
            # print ("Iteration " + str(step*n_batch_size) + ", Minibatch Loss = " + "{:.5f}".format(batch_loss) + \
            #       " [{:.5f}]".format(average_batch_loss) + \
            #       ", Training Accuracy = " + "{:.4f}".format(batch_acc) + \
            #        ", Test Accuracy = " + "{:.4f}".format(test_acc),
            #         "*" if test_acc > best_test_acc else "")
            #
            # saver = tf.train.Saver()
            # saver.save(sess, last_model_file_path)
            #
            # if (test_acc > best_test_acc):
            #     best_test_acc = test_acc
            # saver.save(sess, max_acc_model_file_path)

            prev_output_time = datetime.datetime.now()

        step += 1

    sess.close()