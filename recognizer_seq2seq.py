import sys
from PIL import Image
from toolbox import wordio
import prepare_features as pf
import blstm_seq2seq.rnn_seq2seq_model as model
import tensorflow as tf


def recognize_seq2seq(ds,images_data, image_lengths, word_boxes, text_lines, output_words_file_path, n_rnn_steps):
    # Parameters
    knmp_model_file_path = "./blstm_seq2seq/models/knmp_seq2seq_1x96_1x96_word_acc_0.6848_levenshtein_acc_0.8278.model" # Assume current directory
    stanford_model_file_path = "./blstm_seq2seq/models/knmp_seq2seq_1x96_1x96_word_acc_0.6848_levenshtein_acc_0.8278.model" # Assume current directory
    model_file_path = ""

    knmp_unique_chars = [' ', '!', '#', '%', '&', 'A', 'B', 'C', 'D', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y']
    stanford_unique_chars = [' ', '!', '#', '$', '%', '*', 'B', 'D', 'G', 'H', 'L', 'N', 'O', 'S', 'T', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y']
    unique_chars = []

    if ds == "KNMP":
        model_file_path = knmp_model_file_path
        unique_chars = knmp_unique_chars
    if ds == "STANFORD":
        model_file_path = stanford_model_file_path
        unique_chars = stanford_unique_chars

    # model_file_path = "/Users/rmencis/RUG/Handwriting_Recognition/models/KNMP/last.model" # Assume current directory
    n_image_rnn_steps = n_rnn_steps # Max width of word box
    n_image_features = len(images_data[0][0])
    n_label_rnn_steps = 10 # Max number of characters in label

    n_classes = len(unique_chars)

    print("Model path:",model_file_path)
    print("Maximal number of timesteps:",n_image_rnn_steps)
    print("Features:",n_image_features)
    print("Maximal number of characters in a word:",n_label_rnn_steps)
    print("Classes:",n_classes)

    # Define RNN
    print("Initializing RNN...")
    image_rnn_input_data = tf.placeholder("float", [None, n_image_rnn_steps, n_image_features]) # (n_batch_size, n_steps, n_features)
    image_rnn_input_lengths = tf.placeholder("int32", [None])  # (n_batch_size)
    label_rnn_input_data = tf.placeholder("float", [None, n_label_rnn_steps, n_classes]) # (n_batch_size,n_label_rnn_steps)

    label_rnn_outputs,label_rnn_predicted_index_labels = model.define_seq2seq_rnn_for_prediction(image_rnn_input_data, image_rnn_input_lengths, label_rnn_input_data)

    # Initialize session
    sess = tf.InteractiveSession()

    # Initiale the variables
    init = tf.initialize_all_variables()
    sess.run(init)

    restore_saver = tf.train.Saver()
    restore_saver.restore(sess, model_file_path)

    print("Recognizing...")
    predicted_train_sample_text_labels = model.get_label_rnn_result(label_rnn_predicted_index_labels,
                                                                    image_rnn_input_data, image_rnn_input_lengths,
                                                                    label_rnn_input_data,
                                                                    unique_chars, images_data, image_lengths)

    for word_box,predicted_train_sample_text_label in zip(word_boxes,predicted_train_sample_text_labels):
        predicted_text = predicted_train_sample_text_label.strip()
        print("Word box:",(word_box.left, word_box.top, word_box.right, word_box.bottom),"Correct word:",word_box.text,"Predicted word:",predicted_text)
        word_box.text = predicted_text

    # Save output .words file
    wordio.save(text_lines, output_words_file_path)

    print("Finished.")