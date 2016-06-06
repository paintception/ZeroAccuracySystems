import pandas as pd
import os
import jellyfish


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


if __name__ == "__main__":
    wm = WordM()
    wm.get_closest_word('oech')
    pass