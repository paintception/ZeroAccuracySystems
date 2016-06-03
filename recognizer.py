import argparse
from PIL import Image
from toolbox import wordio
import prepare_features as pf

from recognizer_seq2seq import recognize_seq2seq
import blstm_ctc_net.blstm_ctc_net_model as blstm_ctc_net


def parse_args():
    parser = argparse.ArgumentParser(description='Database initialization')
    group = parser.add_mutually_exclusive_group()

    group.add_argument('-s', '--seq2seq', dest='seq2seq', action='append', nargs=3, metavar=('INPUT_IMAGE', 'INPUT_WORDS', 'OUTPUT_WORDS'),
                       help='Use seq2seq model for recognition')

    group.add_argument('-c', '--ctc', dest='ctc', action='append', nargs=3, metavar=('INPUT_IMAGE', 'INPUT_WORDS', 'OUTPUT_WORDS'),
                       help='Use ctc model for recognition')

    input_args = parser.parse_args()

    if input_args.seq2seq:
        n_image_rnn_steps = 50
        text_lines, word_boxes, images_data, _ = prepare_data(input_args.seq2seq[0][0], input_args.seq2seq[0][1], n_image_rnn_steps)
        recognize_seq2seq(images_data, word_boxes, text_lines, input_args.seq2seq[0][2], n_image_rnn_steps)
    if input_args.ctc:
        n_image_rnn_steps = 100
        text_lines, word_boxes, images_data, images_length = prepare_data(input_args.ctc[0][0], input_args.ctc[0][1], n_image_rnn_steps)
        batch_words = [word_box.text for word_box in word_boxes]
        blstm_ctc_net.blstm_ctc_predict(images_data, images_length, batch_words)



def prepare_data(page_file_path, input_words_file_path, n_image_rnn_steps):
    # Read PPM
    page_image = Image.open(page_file_path)
    print("Page image size:", page_image.width, "x", page_image.height)

    # Read .words file to get word boxes
    text_lines, image_name = wordio.read(input_words_file_path)

    word_boxes = []
    images_data = []
    images_length = []
    for text_line in text_lines:
        for word_box in text_line:
            word_boxes.append(word_box)
            box = (word_box.left, word_box.top, word_box.right, word_box.bottom)
            word_image = page_image.crop(box)
            preprocessed_word_image = pf.preprocess_image(word_image)
            image_data = pf.get_feature_data_for_image(preprocessed_word_image)
            image_data = pf.get_data_with_fixed_time_step_count(image_data, n_image_rnn_steps)
            images_data.append(image_data)
            images_length.append(len(images_data))

    return text_lines, word_boxes, images_data, images_length

if __name__ == "__main__":
    parse_args()