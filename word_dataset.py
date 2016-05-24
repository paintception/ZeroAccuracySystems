import prepare_features as pf
import os
import random
import copy
from PIL import Image

START_WORD_CHAR = "%"

class WordDataItem(object):
    def __init__(self,file_path):
        self._file_path = file_path
        self._label = pf.get_word_label_from_filename(self._file_path)
        self._data = None
        self._width = None

    def get_data(self):
        if self._data is None:
            self._data = pf.get_feature_data(self._file_path)
        return self._data

    def get_data_with_fixed_time_step_count(self,time_step_count):
        tmp_data = copy.copy(self.get_data()) # Weak copy timesteps
        if (len(tmp_data) > time_step_count):
            tmp_data = tmp_data[:time_step_count] # Cut off some timesteps
        else:
            # Add timesteps
            for i in range(self.get_time_step_count(), time_step_count):
                tmp_data.append([0] * self.get_feature_count())
        #return list(reversed(tmp_data))
        return tmp_data

    def get_label(self):
        return self._label

    def get_width(self):
        if self._width is None:
            if self._data is None:
                self._width = Image.open(self._file_path).width
            else:
                self._width = len(self.get_data())
        return self._width

    def get_height(self):
        return len(self.get_data()[0])

    def get_time_step_count(self):
        return self.get_width()

    def get_feature_count(self):
        return self.get_height()

    def get_fixed_length_label(self,fixed_length,start_word_char=False):
        start_char = ""
        if start_word_char is True:
            start_char = START_WORD_CHAR
        tmp_label = start_char + self.get_label()
        if (len(tmp_label) > fixed_length):
            tmp_label = tmp_label[:fixed_length]
        return tmp_label.ljust(fixed_length)

# The class, which keeps dataset (labels, image data etc.) and provides training/test data
class WordDataSet(object):
    def __init__(self, dir_path,max_image_width=1000):
        self._dir_path = dir_path
        self._max_image_width = max_image_width
        self._unique_chars = None

        self.load_data()
        self.init_train_batch()

    def load_data(self):
        self._train_items = self.load_data_items("train")
        self._test_items = self.load_data_items("test")
        self._all_items = self._train_items + self._test_items

    def load_data_items(self,train_vs_test):
        items = []
        file_dir_path = os.path.join(self._dir_path, train_vs_test)
        file_names = [f for f in os.listdir(file_dir_path) if f.endswith(".png")]

        for file_name in file_names:
            file_path = os.path.join(file_dir_path,file_name)

            word_data_item = WordDataItem(file_path)
            if (word_data_item.get_width() <= self._max_image_width):
                items.append(word_data_item)

            if len(items) % 100 == 0:
                print("Loaded %d %s images" % (len(items),train_vs_test))
        print("Loaded all %d %s images" % (len(items),train_vs_test))
        return items

    def get_total_item_count(self):
        return len(self._train_items) + len(self._test_items)

    def get_train_item_count(self):
        return len(self._train_items)

    def get_test_item_count(self):
        return len(self._test_items)

    def get_feature_count(self):
        return self._train_items[0].get_feature_count()

    def init_train_batch(self):
        self._train_items_for_batch = []
        self._next_batch_items = []

    def prepare_next_train_batch(self, batch_size):
        self._next_batch_items = []
        for b in range(batch_size):
            if len(self._train_items_for_batch) == 0:
                self._train_items_for_batch = copy.copy(self._train_items) # Copy only references
                random.shuffle(self._train_items_for_batch)
            train_item = self._train_items_for_batch.pop()
            self._next_batch_items.append(train_item)

    # (n_batch_size,n_time_steps,n_features)
    def get_train_batch_data(self,time_step_count=0):
        return self._get_data(self._next_batch_items, time_step_count)

    # (n_batch_size,n_time_steps,n_features)
    def get_test_data(self, time_step_count=0):
        return self._get_data(self._test_items, time_step_count)

    def _get_data(self,items,time_step_count=0):
        data = []

        for item in items:
            if time_step_count == 0:
                data.append(item.get_data())
            else:
                data.append(item.get_data_with_fixed_time_step_count(time_step_count))
        return data

    # (n_batch_size,n_unique_chars)
    def get_train_batch_first_char_one_hot_labels(self):
        return self._get_first_char_one_hot_labels(self._next_batch_items)

    # (n_batch_size,n_unique_chars)
    def get_test_first_char_one_hot_labels(self):
        return self._get_first_char_one_hot_labels(self._test_items)

    def _get_first_char_one_hot_labels(self,items):
        one_hot_labels = []
        unique_chars = self.get_unique_chars()

        for item in items:
            label = item.get_label()
            char = label[0]
            char_index = unique_chars.index(char)
            one_hot_labels.append(pf.get_one_hot(char_index,len(unique_chars)))
        return one_hot_labels

    def get_train_batch_labels(self):
        return self._get_labels(self._next_batch_items)

    def get_test_labels(self):
        return self._get_labels(self._test_items)

    def _get_labels(self,items):
        labels = []

        for item in items:
            labels.append(item.get_label())
        return labels

    def get_train_batch_fixed_length_one_hot_labels(self, fixed_length, start_word_char=False): # (n_batch_size,n_fixed_length,n_classes)
        return self._get_fixed_length_one_hot_labels(self._next_batch_items, fixed_length, start_word_char)

    def get_test_fixed_length_one_hot_labels(self, fixed_length, start_word_char=False):# (n_test_batch_size,n_fixed_length,n_classes)
        return self._get_fixed_length_one_hot_labels(self._test_items, fixed_length, start_word_char)

    def _get_fixed_length_one_hot_labels(self, items, fixed_length, start_word_char=False):
        one_hot_labels = []

        for item in items:
            label = item.get_fixed_length_label(fixed_length,start_word_char)
            one_hot_label = self._get_one_hot_label(label)
            one_hot_labels.append(one_hot_label)
        return one_hot_labels

    def get_one_hot_labels(self, labels):
        one_hot_labels = []

        for label in labels:
            one_hot_label = self._get_one_hot_label(label)
            one_hot_labels.append(one_hot_label)
        return one_hot_labels

    def get_train_batch_fixed_length_index_labels(self, fixed_length):  # (n_batch_size,n_fixed_length)
        return self._get_fixed_length_index_labels(self._next_batch_items, fixed_length)

    def get_test_fixed_length_index_labels(self, fixed_length):  # (n_test_batch_size,n_fixed_length)
        return self._get_fixed_length_index_labels(self._test_items, fixed_length)

    def _get_fixed_length_index_labels(self, items, fixed_length):
        index_labels = []

        for item in items:
            label = item.get_fixed_length_label(fixed_length)
            index_label = self._get_index_label(label)
            index_labels.append(index_label)
        return index_labels

    def get_unique_chars(self): # List of ['a','A','B',....]
        if self._unique_chars is None:
            chars = [' ',START_WORD_CHAR] # Always add space and start-word character
            for item in self._all_items:
                label = item.get_label()
                for char in label:
                    if not char in chars:
                        chars.append(char)
            self._unique_chars = sorted(chars)
        return self._unique_chars

    def get_max_image_width(self):
        return self.get_max_time_steps()

    def get_max_time_steps(self):
        max_time_steps = 0
        for item in self._all_items:
            if (item.get_time_step_count() > max_time_steps):
                max_time_steps = item.get_time_step_count()
        return max_time_steps


    def get_max_label_length(self):
        max_length = 0
        for item in self._all_items:
            label = item.get_label()
            if (len(label) > max_length):
                max_length = len(label)

        return max_length

    def _get_one_hot_label(self,label):
        unique_chars = self.get_unique_chars()
        one_hot_label = []
        for char in label:
            char_index = unique_chars.index(char)
            one_hot_char = pf.get_one_hot(char_index,len(unique_chars))
            one_hot_label.append(one_hot_char)
        return one_hot_label

    def _get_index_label(self,label): # "abc" => [1,2,3]
        unique_chars = self.get_unique_chars()
        char_index_label = []
        for char in label:
            char_index = unique_chars.index(char)
            char_index_label.append(char_index)
        return char_index_label

    def get_text_labels(self,index_labels):
        unique_chars = self.get_unique_chars()
        text_labels = []
        for index_label in index_labels:
            text_label = ""
            for index in index_label:
                text_label = text_label + unique_chars[index]
            text_labels.append(text_label)
        return text_labels