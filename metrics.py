def get_word_level_accuracy(target_words,predicted_words):
    total_count = 0
    correct_count = 0
    for target_word,predicted_word in zip(target_words,predicted_words):
        total_count += 1
        if target_word == predicted_word:
            correct_count += 1

    return correct_count / total_count

def get_char_level_accuracy(target_words,predicted_words):
    total_count = 0
    correct_count = 0

    for target_word, predicted_word in zip(target_words, predicted_words):
        trimmed_target_word = target_word.strip()

        for target_char, predicted_char in zip(trimmed_target_word,predicted_word):
            total_count += 1
            if target_char == predicted_char:
                correct_count += 1

    return correct_count / total_count