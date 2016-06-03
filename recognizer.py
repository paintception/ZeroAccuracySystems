import sys
from PIL import Image
from toolbox import wordio
import prepare_features as pf
import blstm_seq2seq.rnn_seq2seq_model as model
import tensorflow as tf

def check_parameters():
    if len(sys.argv) < 4:
        print("Please specify command line parameters: input.ppm input.words /path/to/output.words")
        exit()

def get_parameters():
    ppm_file_path = sys.argv[1]
    input_words_file_path = sys.argv[2]
    output_words_file_path = sys.argv[3]
    return ppm_file_path,input_words_file_path,output_words_file_path

# Check command line arguments
check_parameters()

# Get command line arguments
page_file_path, input_words_file_path, output_words_file_path = get_parameters()

print("Page file:", page_file_path)
print("Input labels file:",input_words_file_path)
print("Output labels file:",output_words_file_path)

# Parameters
#model_file_path = "./models/knmp_seq2seq.model" # Assume current directory
model_file_path = "/Users/rmencis/RUG/Handwriting_Recognition/models/KNMP/last.model" # Assume current directory
n_image_rnn_steps = 50 # Max width of word box
n_image_features = pf.FEATURE_COUNT
n_label_rnn_steps = 10 # Max number of characters in label
unique_chars = [' ', '!', '#', '%', '&', 'A', 'B', 'C', 'D', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y']
n_classes = len(unique_chars)

print("Model path:",model_file_path)
print("Maximal number of timesteps:",n_image_rnn_steps)
print("Features:",n_image_features)
print("Maximal number of characters in a word:",n_label_rnn_steps)
print("Classes:",n_classes)

# Read PPM
page_image = Image.open(page_file_path)
print("Page image size:",page_image.width,"x",page_image.height)

# Define RNN
print("Initializing RNN...")
image_rnn_input_data = tf.placeholder("float", [None, n_image_rnn_steps, n_image_features]) # (n_batch_size, n_steps, n_features)
label_rnn_input_data = tf.placeholder("float", [None, n_label_rnn_steps, n_classes]) # (n_batch_size,n_label_rnn_steps)

label_rnn_outputs,label_rnn_predicted_index_labels = model.define_seq2seq_rnn_for_prediction(image_rnn_input_data,label_rnn_input_data)

# Initialize session
sess = tf.InteractiveSession()

# Initiale the variables
init = tf.initialize_all_variables()
sess.run(init)

restore_saver = tf.train.Saver()
restore_saver.restore(sess, model_file_path)

# Read .words file to get word boxes
text_lines, image_name = wordio.read(input_words_file_path)

word_boxes = []
images_data = []
for text_line in text_lines:
    for word_box in text_line:
        word_boxes.append(word_box)
        box = (word_box.left, word_box.top, word_box.right, word_box.bottom)
        word_image = page_image.crop(box)
        preprocessed_word_image = pf.preprocess_image(word_image)
        image_data = pf.get_feature_data_for_image(preprocessed_word_image)
        image_data = pf.get_data_with_fixed_time_step_count(image_data, n_image_rnn_steps)
        images_data.append(image_data)

print("Recognizing...")
predicted_train_sample_text_labels = model.get_label_rnn_result(label_rnn_predicted_index_labels,
                                                                image_rnn_input_data, label_rnn_input_data,
                                                                unique_chars, images_data)

for word_box,predicted_train_sample_text_label in zip(word_boxes,predicted_train_sample_text_labels):
    predicted_text = predicted_train_sample_text_label.strip()
    print("Word box:",(word_box.left, word_box.top, word_box.right, word_box.bottom),"Correct word:",word_box.text,"Predicted word:",predicted_text)
    word_box.text = predicted_text

# Save output .words file
wordio.save(text_lines, output_words_file_path)

print("Finished.")