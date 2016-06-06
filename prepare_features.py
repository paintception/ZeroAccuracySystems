from PIL import Image
from copy import copy, deepcopy
import os
import random

START_WORD_CHAR = "%"

# Return 2D list (time_steps,features). Time steps = image width, Features = pixel values of 1px slices
def get_feature_data_for_image(image, min_time_steps=0):
    # Get pixel values
    pixel_value_sum = 0
    time_steps_with_features = []  # 1px slices
    for x in range(image.width):
        time_step_features = []  # Pixel values of 1px slice (from top to down)
        for y in range(image.height):
            pixel = image.getpixel((x, y))
            pixel_value = 0
            if (pixel[1] != 0):
                pixel_value = pixel[0] / pixel[1]  # Pixel value in range 0..1
            pixel_value_sum = pixel_value_sum + pixel_value
            time_step_features.append(pixel_value)
        time_steps_with_features.append(time_step_features)

    # Normalize around 0
    mean_pixel_value = pixel_value_sum / (image.width * image.height)
    for ts in range(len(time_steps_with_features)):
        time_step_features = time_steps_with_features[ts]
        for f in range(len(time_step_features)):
            time_step_features[f] = time_step_features[f] - mean_pixel_value

    # Pad with empty time steps
    if (min_time_steps > 0):
        for i in range(image.width, min_time_steps):
            # time_step_features = []
            # for j in range(image.height):
            #     time_step_features.append(random.random()-0.5)
            # time_steps_with_features.append(time_step_features)
            time_steps_with_features.append([0] * image.height)

    return time_steps_with_features


def get_feature_data_for_file(image_file_path, min_time_steps=0):
    image = Image.open(image_file_path)
    return get_feature_data_for_image(image, min_time_steps)


def get_data_with_fixed_time_step_count(data, fixed_time_step_count):
    time_step_count = len(data)
    feature_count = len(data[0])
    tmp_data = copy(data)  # Weak copy timesteps
    if (time_step_count > fixed_time_step_count):
        tmp_data = tmp_data[:fixed_time_step_count]  # Cut off some timesteps
    else:
        # Add timesteps
        for i in range(time_step_count, fixed_time_step_count):
            tmp_data.append([0] * feature_count)
    # return list(reversed(tmp_data))
    return tmp_data


# Return class names, where each sub-directory name = class name
def get_classes(label_dir_path):
    class_names = sorted([f for f in os.listdir(label_dir_path) if os.path.isdir(os.path.join(label_dir_path, f)) and not f.startswith(".")])
    return class_names


def get_one_hot(index, length):
    one_hot = [0] * length
    one_hot[index] = 1
    return one_hot


def get_one_hot_label(unique_chars, label):
    one_hot_label = []
    for char in label:
        char_index = unique_chars.index(char)
        one_hot_char = get_one_hot(char_index, len(unique_chars))
        one_hot_label.append(one_hot_char)
    return one_hot_label


def get_one_hot_labels(unique_chars, labels):
    one_hot_labels = []

    for label in labels:
        one_hot_label = get_one_hot_label(unique_chars, label)
        one_hot_labels.append(one_hot_label)
    return one_hot_labels


def get_word_label_from_filename(file_name):
    label = file_name.replace(".png", "")
    last_sep = label.rfind("_")
    label = label[last_sep + 1:]
    return label


def get_text_labels(unique_chars, index_labels):
    text_labels = []
    for index_label in index_labels:
        text_label = get_text_label(unique_chars, index_label)
        text_labels.append(text_label)
    return text_labels


def get_text_label(unique_chars, index_label):
    text_label = ""
    for index in index_label:
        text_label = text_label + unique_chars[index]
    return text_label


def get_index_label(unique_chars, label):  # "abc" => [1,2,3]
    char_index_label = []
    for char in label:
        char_index = unique_chars.index(char)
        char_index_label.append(char_index)
    return char_index_label


def get_fixed_length_label(label, fixed_length, start_word_char=False):
    start_char = ""
    if start_word_char is True:
        start_char = START_WORD_CHAR
    tmp_label = start_char + label
    if (len(tmp_label) > fixed_length):
        tmp_label = tmp_label[:fixed_length]
    return tmp_label.ljust(fixed_length)


# Takes original word box and returns word box for NN training/predicting
def preprocess_image(image, feature_count=16, resize_ratio=0.25):
    image = image.convert("LA")  # Greyscale
    new_width = int(image.width * resize_ratio)
    new_height = feature_count
    return image.resize((new_width, new_height), Image.LANCZOS)
