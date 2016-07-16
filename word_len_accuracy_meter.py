import os
from toolbox import wordio
import metrics

predicted_dir_path = "/Users/rmencis/RUG/Handwriting_Recognition/output_words/ctc"
target_dir_path = "/Users/rmencis/RUG/Handwriting_Recognition/code/ZeroAccuracySystems/labels"

words_file_names = [f for f in os.listdir(predicted_dir_path) if f.endswith(".words")]

total = {}
correct = {}
total_sum = 0
correct_sum = 0

predicted_words = []
target_words = []

for wl in range(21):
    total[wl] = 0
    correct[wl] = 0

for words_file_name in words_file_names:
    predicted_words_file_path = os.path.join(predicted_dir_path,words_file_name)
    target_words_file_path = os.path.join(target_dir_path,words_file_name)

    # Predicted
    predicted_lines, predicted_image_name = wordio.read(predicted_words_file_path)
    for predicted_line in predicted_lines:
        for word in predicted_line:
            predicted_words.append(word.text)

    # Target
    target_lines, target_image_name = wordio.read(target_words_file_path)
    for target_line in target_lines:
        for word in target_line:
            target_words.append(word.text)

for predicted_word, target_word in zip(predicted_words, target_words):
    total[len(target_word)] += 1
    total_sum += 1
    if predicted_word == target_word:
        correct[len(target_word)] += 1
        correct_sum += 1

for wl in range(1,21):
    acc = 0
    if total[wl] > 0:
        acc = correct[wl] / total[wl]
    print("Word length:", wl, "Total:",total[wl],"Correct:",correct[wl],"Accuracy:","{:.2f}%".format(acc*100))

print("ALL Total:", total_sum, "Correct:", correct_sum, "Accuracy:", "{:.2f}%".format(correct_sum/total_sum * 100))

