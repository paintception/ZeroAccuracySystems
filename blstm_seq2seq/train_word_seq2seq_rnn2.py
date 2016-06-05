import datetime
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import blstm_seq2seq.word_dataset as wd
from blstm_seq2seq.word_dataset import WordDataSetRM,WordDataItemRM
import dirs
import random
import os
from tensorflow.python.ops.seq2seq import sequence_loss
import metrics
import blstm_seq2seq.rnn_seq2seq_model as model
import prepare_features as pf

# Saved models
model_dir_path = dirs.STANFORD_MODEL_DIR_PATH
last_model_file_path = os.path.join(model_dir_path,"last.model")
max_acc_model_file_path = os.path.join(model_dir_path,"max_acc.model")
if not os.path.exists(model_dir_path):
    os.makedirs(model_dir_path)

# Read data set
fixed_timestep_count = 50
max_image_width = 50
dataset = WordDataSetRM(dirs.STANFORD_PROCESSED_WORD_BOXES_DIR_PATH,max_image_width=max_image_width)

print("Total items:",dataset.get_total_item_count())
print("Training items:",dataset.get_train_item_count())
n_test_items = dataset.get_test_item_count()
print("Test items:",n_test_items)
print("Max time steps (width):",dataset.get_max_time_steps())
n_label_rnn_steps = dataset.get_max_label_length() + 1
print("Max label length:", n_label_rnn_steps)

# Parameters
learning_rate = 0.1
print("Learning rate:",learning_rate)
n_batch_size = 256
print("Batch size:",n_batch_size)
dropout_input_keep_prob_value = 0.75
print('Dropout input keep probability:',dropout_input_keep_prob_value)
dropout_output_keep_prob_value = 0.75
print('Dropout output keep probability:',dropout_output_keep_prob_value)

n_classes = len(dataset.get_unique_chars()) # Classes (A,a,B,b,c,...)
print("Classes:", n_classes)
n_image_features = dataset.get_feature_count() # Features = image height
print(dataset.get_unique_chars())
print("Features:", n_image_features)
n_image_rnn_steps = fixed_timestep_count # Timesteps = image width
print("Time steps:", n_image_rnn_steps)
display_time_interval_sec = 30

# Placeholders
default_dropout_prob = tf.constant(1,"float")
dropout_input_keep_prob = tf.placeholder_with_default(default_dropout_prob,[])
dropout_output_keep_prob = tf.placeholder_with_default(default_dropout_prob,[])
image_rnn_input_data = tf.placeholder("float", [None, n_image_rnn_steps, n_image_features]) # (n_batch_size, n_steps, n_features)
label_rnn_input_data = tf.placeholder("float", [None, n_label_rnn_steps, n_classes])
label_rnn_target_data = tf.placeholder("int64", [None, n_label_rnn_steps]) # (n_batch_size,n_label_rnn_steps)

# Define RNN

label_rnn_outputs,label_rnn_predicted_index_labels = model.define_seq2seq_rnn_for_training(image_rnn_input_data,label_rnn_input_data,dropout_input_keep_prob,dropout_output_keep_prob)

# Transform target data for label RNN
label_rnn_target_outputs = tf.transpose(label_rnn_target_data, [1, 0]) # (n_label_rnn_steps,n_batch_size)
label_rnn_target_outputs = tf.split(0,n_label_rnn_steps,label_rnn_target_outputs)
label_rnn_target_outputs = [tf.squeeze(lrt) for lrt in label_rnn_target_outputs]

# Optimization
weights_shape = tf.shape(label_rnn_target_outputs[0])
sequence_loss_weights = [tf.ones(weights_shape)]*n_label_rnn_steps
sequence_loss_weight_value = 10.0
# # Higher weights for first characters 10,5,2.5,1.25,...
for i in range(len(sequence_loss_weights)):
    sequence_loss_weights[i] = tf.fill(weights_shape,sequence_loss_weight_value)
    sequence_loss_weight_value = sequence_loss_weight_value * 0.5
    if sequence_loss_weight_value < 1:
        break
cost = sequence_loss(label_rnn_outputs,label_rnn_target_outputs,sequence_loss_weights)

#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer2 = tf.train.GradientDescentOptimizer(learning_rate)

optimizer_step = optimizer2.minimize(cost)  # Adam Optimizer
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,epsilon=0.1,beta1=0.9, beta2=0.999).minimize(cost)  # Adam Optimizer

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

    while True:
        # Training
        dataset.prepare_next_train_batch(n_batch_size)
        train_batch_data = dataset.get_train_batch_data(time_step_count=fixed_timestep_count)  # (batch_size,n_steps,n_input)
        train_batch_one_hot_labels = dataset.get_train_batch_fixed_length_one_hot_labels(n_label_rnn_steps, start_word_char=True) # (batch_size,n_output_steps,n_classes)
        train_batch_index_labels = dataset.get_train_batch_fixed_length_index_labels(n_label_rnn_steps) # (batch_size,n_output_steps)

        sess.run(optimizer_step, feed_dict={image_rnn_input_data:train_batch_data,
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

            predicted_train_sample_text_labels = model.get_label_rnn_result(label_rnn_predicted_index_labels,image_rnn_input_data,label_rnn_input_data,dataset.get_unique_chars(),train_sample_data)
            train_sample_word_level_accuracy = metrics.get_word_level_accuracy(train_sample_text_labels, predicted_train_sample_text_labels)
            train_sample_char_level_accuracy = metrics.get_char_level_accuracy(train_sample_text_labels, predicted_train_sample_text_labels)
            train_sample_avg_word_distance = metrics.get_avg_word_distance(train_sample_text_labels, predicted_train_sample_text_labels)

            predicted_test_text_labels = model.get_label_rnn_result(label_rnn_predicted_index_labels,image_rnn_input_data,label_rnn_input_data,dataset.get_unique_chars(),test_data)
            test_word_level_accuracy = metrics.get_word_level_accuracy(test_text_labels,predicted_test_text_labels)
            test_char_level_accuracy = metrics.get_char_level_accuracy(test_text_labels,predicted_test_text_labels)
            test_avg_word_distance = metrics.get_avg_word_distance(test_text_labels, predicted_test_text_labels)

            print("Iter:", step * n_batch_size, "TRAINING Loss: " + "{:.5f}".format(train_sample_loss), "[{:.5f}]".format(average_batch_loss), "Word acc.: " + "{:.4f}".format(train_sample_word_level_accuracy), "Char acc.: " + "{:.4f}".format(train_sample_char_level_accuracy), "Levenstein: " + "{:.4f}".format(train_sample_avg_word_distance),
                  "TEST","Word acc.: " + "{:.4f}".format(test_word_level_accuracy), "Char acc.: " + "{:.4f}".format(test_char_level_accuracy), "Levenstein: " + "{:.4f}".format(test_avg_word_distance),
                  "*" if test_avg_word_distance > best_accuracy else "")
            max_output_items = 15
            print("TRAINING SAMPLE:")
            print(train_sample_text_labels[:max_output_items])
            print(predicted_train_sample_text_labels[:max_output_items])
            print("TEST:")
            print(test_text_labels[:max_output_items])
            print(predicted_test_text_labels[:max_output_items])
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
