import prepare_features
import datetime
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
from dataset import DataSet
import dirs

dataset = DataSet(dirs.KNMP_PROCESSED_CHAR_BOXES_DIR_PATH)

print("Total items:",dataset.get_total_item_count())
print("Training items:",dataset.get_train_item_count())
print("Test items:",dataset.get_test_item_count())

# Parameters
learning_rate = 0.0005
print("Learning rate:",learning_rate)
batch_size = 256
print("Batch size:",batch_size)
display_time = 5

dropout_input_keep_prob = tf.placeholder("float")
dropout_input_keep_prob_value = 0.5
print('Dropout input keep probability:',dropout_input_keep_prob_value)

dropout_output_keep_prob = tf.placeholder("float")
dropout_output_keep_prob_value = 0.5
print('Dropout output keep probability:',dropout_output_keep_prob_value)

# Network Parameters
n_input = dataset.get_feature_count() # Features = image height
print("Features:",n_input)
n_steps = dataset.get_time_step_count() # Timesteps = image width
print("Time steps:",n_steps)
n_hidden = 24 # hidden layer num of features
print("Hidden units:",n_hidden)
n_classes = dataset.get_class_count() # Classes (A,a,B,b,c,...)
print("Classes:",n_classes)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
istate = tf.placeholder("float", [None, 2*n_hidden]) #state & cell => 2x n_hidden
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(_X, _istate, _weights, _biases):

    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size => (n_steps,batch_size,n_input)
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input) (2D list with 28*50 vectors with 28 features each)
    # Linear activation
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden'] # (n_steps*batch_size=28*50,n_hidden=128)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    lstm_cell_drop = rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=dropout_input_keep_prob, output_keep_prob=dropout_output_keep_prob)

    #multi_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2)

    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden) => step1 (batch_size=128,n_hidden=128)..step28 (batch_size=128,n_hidden=128)
    # It means that RNN receives list with element (batch_size,n_hidden) for each time step

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell_drop, _X, initial_state=_istate)
    # Output is list with element (batch_size,n_hidden) for each time step?
    #for output in outputs:
    #    print(output)
    #exit(0)

    # Linear activation
    # Get inner loop last output
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out']

pred = RNN(x, istate, weights, biases) # I guess it is (batch_size,n_classes)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    prev_output_time = datetime.datetime.now()
    best_test_acc = 0
    batch_losses = []

    while True:
        #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        dataset.prepare_next_batch(batch_size)
        batch_xs = dataset.get_batch_data() # (batch_size,n_steps,n_input)
        batch_ys = dataset.get_batch_one_hot_labels() # (batch_size,n_classes)

        # Reshape data to get 28 seq of 28 elements
        #batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                       istate: np.zeros((batch_size, 2*n_hidden)),
                                            dropout_input_keep_prob: dropout_input_keep_prob_value,
                                                dropout_output_keep_prob: dropout_output_keep_prob_value})

        from_prev_output_time = datetime.datetime.now() - prev_output_time
        if step == 1 or from_prev_output_time.seconds > display_time:
            # Calculate training batch accuracy
            batch_acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
                                                istate: np.zeros((batch_size, 2*n_hidden)),
                                                    dropout_input_keep_prob: dropout_input_keep_prob_value,
                                                        dropout_output_keep_prob: dropout_output_keep_prob_value})
            # Calculate training batch loss
            batch_loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,
                                             istate: np.zeros((batch_size, 2*n_hidden)),
                                                dropout_input_keep_prob: dropout_input_keep_prob_value,
                                                    dropout_output_keep_prob: dropout_output_keep_prob_value})
            batch_losses.append(batch_loss)
            avg_count = 10
            last_batch_losses = batch_losses[-min(avg_count, len(batch_losses)):]
            average_batch_loss = sum(last_batch_losses) / len(last_batch_losses)

            # Calculate test accuracy
            test_xs = dataset.get_test_data()
            test_ys = dataset.get_test_one_hot_labels()

            test_acc = sess.run(accuracy, feed_dict={x: test_xs, y: test_ys,
                                                      istate: np.zeros((len(test_xs), 2 * n_hidden)),
                                       dropout_input_keep_prob: 1.0,
                                        dropout_output_keep_prob: 1.0})
            print ("Iteration " + str(step*batch_size) + ", Minibatch Loss = " + "{:.4f}".format(batch_loss) + \
                  " [{:.4f}]".format(average_batch_loss) + \
                  ", Training Accuracy = " + "{:.4f}".format(batch_acc) + \
                   ", Test Accuracy = " + "{:.4f}".format(test_acc),
                    "*" if test_acc > best_test_acc else "")

            if (test_acc > best_test_acc):
                best_test_acc = test_acc

            prev_output_time = datetime.datetime.now()
        step += 1
