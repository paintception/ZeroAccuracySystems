import pandas as pd
import os
import jellyfish
import pickle

from blstm_ctc_net import dirs


class WordM:

    word_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "hwr_course_lexicon_testset.txt")
    small_word_threshold = 1

    def __init__(self):
        self.words = pd.read_csv(WordM.word_file, delimiter=' ', names=['word', 'frequency'], header=None)


    def get_distances(self, input_word):
        return self.words['word'].apply(lambda lw: jellyfish.levenshtein_distance(lw, input_word) / max(len(lw), len(input_word)))


    def get_closest_word(self, input_word):
        if len(input_word) <= WordM.small_word_threshold:
            return input_word

        distances = self.get_distances(input_word)

        return self.words[distances == distances.min()].sort_values(by='frequency', ascending=False).reset_index().loc[0, 'word']


class WordMOL:

    word_file_KNMP = os.path.join(dirs.KNMP_WORD_BOXES_DIR_PATH, "word_info")
    word_file_STANFORD = os.path.join(dirs.STANFORD_WORD_BOXES_DIR_PATH, "word_info")
    small_word_threshold = 1

    def __init__(self, dataset):
        word_file = WordMOL.word_file_KNMP if dataset == 'KNMP' else WordMOL.word_file_STANFORD
        with open(word_file, 'rb') as word_info_file:
            temp_df = pd.DataFrame.from_dict(pickle.load(word_info_file))

        temp_df['word'] = temp_df['char_labels'].str.join('')

        temp_df.drop(['box_image_name', 'char_labels', 'char_positions', 'word_length'], axis=1, inplace=True)

        self.words = temp_df['word'].value_counts().reset_index(name='word')

        self.words.rename(columns={'word': 'frequency', 'index': 'word'}, inplace=True)

        pass


    def get_distances(self, input_word):
        return self.words['word'].apply(lambda lw: jellyfish.levenshtein_distance(lw, input_word) / max(len(lw), len(input_word)))


    def get_closest_word(self, input_word):
        if len(input_word) <= WordM.small_word_threshold:
            return input_word

        distances = self.get_distances(input_word)

        return self.words[distances == distances.min()].sort_values(by='frequency', ascending=False).reset_index().loc[0, 'word']


if __name__ == "__main__":
    wm = WordMOL()
    pass