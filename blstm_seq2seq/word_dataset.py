import prepare_features as pf
import os
import random
import copy
from PIL import Image

class WordDataItemRM(object):
    def __init__(self,file_path):
        self._file_path = file_path
        self._label = pf.get_word_label_from_filename(self._file_path)
        self._data = None
        self._width = None

    def get_data(self):
        if self._data is None:
            self._data = pf.get_feature_data_for_file(self._file_path)
        return self._data

    def get_data_with_fixed_time_step_count(self,time_step_count):
        return pf.get_data_with_fixed_time_step_count(self.get_data(),time_step_count)

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
        return pf.get_fixed_length_label(self.get_label(),fixed_length,start_word_char)

# The class, which keeps dataset (labels, image data etc.) and provides training/test data
class WordDataSetRM(object):
    def __init__(self, dir_path,max_image_width=1000):
        self._dir_path = dir_path
        self._max_image_width = max_image_width
        self._unique_chars = None
        self._all_lengths = None

        self.load_data()
        self.init_train_batch()

    def load_data(self):
        self._train_items = self.load_data_items("train")
        self._test_items = self.load_data_items("test")
        random.shuffle(self._test_items)
        self._all_items = self._train_items + self._test_items

    def load_data_items(self,train_vs_test):
        items = []
        file_dir_path = os.path.join(self._dir_path, train_vs_test)
        file_names = [f for f in os.listdir(file_dir_path) if f.endswith(".png")]

        for file_name in file_names:
            file_path = os.path.join(file_dir_path,file_name)

            word_data_item = WordDataItemRM(file_path)
            if (word_data_item.get_width() <= self._max_image_width):
                items.append(word_data_item)

            if len(items) % 1000 == 0:
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

    def prepare_balanced_next_train_batch(self, batch_size,time_step_count=9999):
        all_lengths = self.get_all_sequence_lengths(time_step_count)
        # interval_start_index = random.randrange(len(all_lengths))
        # interval_start = all_lengths[interval_start_index]
        # interval_end_index = interval_start_index + batch_size - 1
        # if interval_end_index >= len(all_lengths):
        #     interval_end_index = len(all_lengths)-1
        # interval_end = all_lengths[interval_end_index]

        interval_end = random.sample(all_lengths,1)[0]
        self.prepare_next_train_batch(batch_size,(0,interval_end))

    def prepare_next_train_batch(self, batch_size, length_interval=(0,9999)):
        self._next_batch_items = []
        counter = 0
        for b in range(len(self._train_items)):
            if len(self._train_items_for_batch) == 0:
                self._train_items_for_batch = copy.copy(self._train_items) # Copy only references
                random.shuffle(self._train_items_for_batch)
            train_item = self._train_items_for_batch.pop()
            if train_item.get_time_step_count() >= length_interval[0] and train_item.get_time_step_count() <= length_interval[1]:
                if not train_item in self._next_batch_items:
                    self._next_batch_items.append(train_item)
            if len(self._next_batch_items) == batch_size:
                break
        #print("Length interval:", length_interval,"Batch size:",len(self._next_batch_items))

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

    def get_train_batch_sequence_lengths(self, time_step_count=None):
        return self._get_sequence_length(self._next_batch_items, time_step_count)

    def get_test_sequence_lengths(self, time_step_count=None):
        return self._get_sequence_length(self._test_items, time_step_count)

    def get_all_sequence_lengths(self, time_step_count=None):
        if self._all_lengths is None:
            self._all_lengths = sorted(self._get_sequence_length(self._all_items, time_step_count))
        return self._all_lengths

    def _get_sequence_length(self, items, time_step_count=None):
        if time_step_count:
            return [item.get_time_step_count() if item.get_time_step_count() < time_step_count else time_step_count
                    for item in items]
        else:
            return [item.get_time_step_count() for item in items]

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
            one_hot_label = pf.get_one_hot_label(self.get_unique_chars(),label)
            one_hot_labels.append(one_hot_label)
        return one_hot_labels

    def get_one_hot_labels(self, labels):
        return pf.get_one_hot_labels(self.get_unique_chars(),labels)

    def get_train_batch_fixed_length_index_labels(self, fixed_length):  # (n_batch_size,n_fixed_length)
        return self._get_fixed_length_index_labels(self._next_batch_items, fixed_length)

    def get_test_fixed_length_index_labels(self, fixed_length):  # (n_test_batch_size,n_fixed_length)
        return self._get_fixed_length_index_labels(self._test_items, fixed_length)

    def _get_fixed_length_index_labels(self, items, fixed_length):
        index_labels = []

        for item in items:
            label = item.get_fixed_length_label(fixed_length)
            index_label = pf.get_index_label(self.get_unique_chars(),label)
            index_labels.append(index_label)
        return index_labels

    def get_unique_chars(self): # List of ['a','A','B',....]
        if self._unique_chars is None:
            chars = [' ',pf.START_WORD_CHAR] # Always add space and start-word character
            for item in self._all_items:
                label = item.get_label()
                for char in label:
                    if not char in chars:
                        chars.append(char)
            self._unique_chars = sorted(chars)
        return self._unique_chars

    # def get_unique_chars_as_string(self):
    #     unique_chars = self.get_unique_chars()
    #     for unique_char in unique_chars:
    #

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

    # def _get_one_hot_label(self,label):
    #     unique_chars = self.get_unique_chars()
    #     one_hot_label = []
    #     for char in label:
    #         char_index = unique_chars.index(char)
    #         one_hot_char = pf.get_one_hot(char_index,len(unique_chars))
    #         one_hot_label.append(one_hot_char)
    #     return one_hot_label

    # def _get_index_label(self,label): # "abc" => [1,2,3]
    #     unique_chars = self.get_unique_chars()
    #     char_index_label = []
    #     for char in label:
    #         char_index = unique_chars.index(char)
    #         char_index_label.append(char_index)
    #     return char_index_label

    def get_text_labels(self,index_labels):
        return pf.get_text_labels(self.get_unique_chars(),index_labels)
