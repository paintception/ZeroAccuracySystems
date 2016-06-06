import sys
from PIL import Image
from toolbox import wordio
import prepare_features as pf
import blstm_seq2seq.rnn_seq2seq_model as model
import tensorflow as tf
from tensorflow.python.ops.seq2seq import sequence_loss
import metrics

def recognize_seq2seq(ds,images_data, image_lengths, word_boxes, n_rnn_steps):
    if ds == "KNMP":
        model_file_path = "./blstm_seq2seq/models/knmp_seq2seq_1x96_1x96_word_acc_0.6848_levenshtein_acc_0.8278.model" # Assume current directory
        unique_chars = [' ', '!', '#', '%', '&', 'A', 'B', 'C', 'D', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y']
        n_label_rnn_steps = 10
    if ds == "STANFORD":
        model_file_path = "./blstm_seq2seq/models/stanford_seq2seq_1x96_1x96_word_acc_0.6739_levenshtein_acc_0.7587.model"
        unique_chars = [' ', '!', '#', '$', '%', '*', 'B', 'D', 'G', 'H', 'L', 'N', 'O', 'S', 'T', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y']
        n_label_rnn_steps = 9

    # model_file_path = "/Users/rmencis/RUG/Handwriting_Recognition/models/KNMP/last.model" # Assume current directory
    n_image_rnn_steps = n_rnn_steps # Max width of word box
    n_image_features = len(images_data[0][0])

    n_classes = len(unique_chars)

    print("Model path:",model_file_path)
    print("Maximal number of timesteps:",n_image_rnn_steps)
    print("Features:",n_image_features)
    print("Maximal number of characters in a word:",n_label_rnn_steps)
    print("Classes:",n_classes)

    # Define RNN
    print("Initializing RNN...")
    image_rnn_input_data = tf.placeholder("float", [None, n_image_rnn_steps,
                                                    n_image_features])  # (n_batch_size, n_steps, n_features)
    image_rnn_input_lengths = tf.placeholder("int32", [None])  # (n_batch_size)
    label_rnn_input_data = tf.placeholder("float", [None, n_label_rnn_steps, n_classes])

    label_rnn_outputs,label_rnn_predicted_index_labels = model.define_seq2seq_rnn_for_prediction(image_rnn_input_data, image_rnn_input_lengths, label_rnn_input_data)

    # Initialize session
    sess = tf.InteractiveSession()

    # Initiale the variables
    init = tf.initialize_all_variables()
    sess.run(init)

    restore_saver = tf.train.Saver()
    restore_saver.restore(sess, model_file_path)

    print("Recognizing...")
    predicted_labels = model.get_label_rnn_result(label_rnn_predicted_index_labels,
                                                                    image_rnn_input_data, image_rnn_input_lengths,
                                                                    label_rnn_input_data,
                                                                    unique_chars, images_data, image_lengths)

    predicted_labels = [label.strip() for label in predicted_labels]
    real_labels = [word_box.text.strip() for word_box in word_boxes]

    word_accuracy = metrics.get_word_level_accuracy(real_labels,predicted_labels)
    char_accuracy = metrics.get_char_level_accuracy(real_labels,predicted_labels)
    levenshtein_accuracy = metrics.get_avg_word_distance(real_labels,predicted_labels)

    print("Word accuracy:","{:.4f}".format(word_accuracy),"Char accuracy:","{:.4f}".format(char_accuracy),"Levenshtein accuracy:","{:.4f}".format(levenshtein_accuracy))

    print([label.ljust(15) for label in real_labels])
    print([label.ljust(15) for label in predicted_labels])

    return predicted_labels

    # for word_box,predicted_train_sample_text_label in zip(word_boxes,predicted_train_sample_text_labels):
    #     predicted_text = predicted_train_sample_text_label.strip()
    #     print("Word box:",(word_box.left, word_box.top, word_box.right, word_box.bottom),"Correct word:",word_box.text,"Predicted word:",predicted_text)
    #     word_box.text = predicted_text
    #
    #
    #
    # print(predicted_train_sample_text_labels)
    #
    # # Save output .words file
    # wordio.save(text_lines, output_words_file_path)
    #
    # print("Finished.")