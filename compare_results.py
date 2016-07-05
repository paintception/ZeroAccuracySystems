import os
from toolbox import wordio
import metrics

ctc_dir_path = "/Users/rmencis/RUG/Handwriting_Recognition/subsettest/output/ctc"
s2s_dir_path = "/Users/rmencis/RUG/Handwriting_Recognition/subsettest/output/seq2seq"

words_file_names = [f for f in os.listdir(ctc_dir_path) if f.endswith(".words")]

word_count = 0
equal_word_count = 0
ctc_words = []
s2s_words = []
for words_file_name in words_file_names:
    ctc_words_file_path = os.path.join(ctc_dir_path,words_file_name)
    s2s_words_file_path = os.path.join(s2s_dir_path,words_file_name)

    # CTC
    ctc_lines, ctc_image_name = wordio.read(ctc_words_file_path)
    for ctc_line in ctc_lines:
        for word in ctc_line:
            ctc_words.append(word.text)

    # Seq2seq
    s2s_lines, s2s_image_name = wordio.read(s2s_words_file_path)
    for s2s_line in s2s_lines:
        for word in s2s_line:
            s2s_words.append(word.text)

for ctc_word, s2s_word in zip(ctc_words, s2s_words):
    word_count = word_count + 1
    if ctc_word == s2s_word:
        equal_word_count = equal_word_count + 1

print("Word count:",word_count)
print("Equal word count:",equal_word_count)
print("Equal word ratio:",equal_word_count/word_count)
print("Levenshtein:",metrics.get_avg_word_distance(ctc_words,s2s_words))
