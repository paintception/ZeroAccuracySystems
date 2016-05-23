def get_word_level_accuracy(target_words,test_words):
    total_count = 0
    correct_count = 0
    for target_word,test_word in zip(target_words,test_words):
        total_count += 1
        if target_word == test_word:
            correct_count += 1

    return correct_count / total_count

def get_char_level_accuracy(target_words,test_words):
    total_count = 0
    correct_count = 0

    for target_word, test_word in zip(target_words, test_words):
        trimmed_target_word = target_word.strip()

        for target_char, test_char in zip(trimmed_target_word,test_word):
            total_count += 1
            if target_char == test_char:
                correct_count += 1

    return correct_count / total_count