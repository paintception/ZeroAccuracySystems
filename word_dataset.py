import prepare_features as pf
import os
import random
import copy

class WordDataItem(object):
    def __init__(self,file_path):
        self._file_path = file_path
        self._label = pf.get_word_label_from_filename(self._file_path)
        self._data = None

    def load_data(self):
        self._data = pf.get_feature_data(self._file_path)

    def get_data(self):
        if self._data == None:
            self.load_data()
        return self._data

    def get_data_with_fixed_time_step_count(self,time_step_count):
        tmp_data = copy.copy(self.get_data()) # Weak copy timesteps
        if (len(tmp_data) > time_step_count):
            tmp_data = tmp_data[:time_step_count] # Cut off some timesteps
        else:
            # Add timesteps
            for i in range(self.get_time_step_count(), time_step_count):
                tmp_data.append([0] * self.get_feature_count())
        return tmp_data

    def get_label(self):
        return self._label

    def get_width(self):
        return len(self.get_data())

    def get_height(self):
        return len(self.get_data()[0])

    def get_time_step_count(self):
        return self.get_width()

    def get_feature_count(self):
        return self.get_height()


# The class, which keeps dataset (labels, image data etc.) and provides training/test data
class WordDataSet(object):
    def __init__(self, dir_path,max_image_width=1000):
        self.dir_path = dir_path
        self.max_image_width = max_image_width

        self.load_data()
        self.init_train_batch()

    def load_data(self):
        self.train_items = self.load_data_items("train")
        self.test_items = self.load_data_items("test")
        self.all_items = self.train_items + self.test_items

    def load_data_items(self,train_vs_test):
        items = []
        file_dir_path = os.path.join(self.dir_path,train_vs_test)
        file_names = [f for f in os.listdir(file_dir_path) if f.endswith(".png")]

        for file_name in file_names:
            file_path = os.path.join(file_dir_path,file_name)

            word_data_item = WordDataItem(file_path)
            #if (word_data_item.get_width() <= self.max_image_width):
            items.append(word_data_item)

            if len(items) % 100 == 0:
                print("Loaded %d %s images" % (len(items),train_vs_test))
        print("Loaded all %d %s images" % (len(items),train_vs_test))
        return items

    def get_total_item_count(self):
        return len(self.train_items) + len(self.test_items)

    def get_train_item_count(self):
        return len(self.train_items)

    def get_test_item_count(self):
        return len(self.test_items)

    def get_feature_count(self):
        return self.train_items[0].get_feature_count()

    def init_train_batch(self):
        self.train_items_for_batch = []
        self.next_batch_items = []

    def prepare_next_train_batch(self, batch_size):
        self.next_batch_items = []
        for b in range(batch_size):
            if len(self.train_items_for_batch) == 0:
                self.train_items_for_batch = copy.copy(self.train_items) # Copy only references
                random.shuffle(self.train_items_for_batch)
            train_item = self.train_items_for_batch.pop()
            self.next_batch_items.append(train_item)

    # (n_batch_size,n_time_steps,n_features)
    def get_train_batch_data(self,time_step_count=0):
        return self._get_data(self.next_batch_items,time_step_count)

    # (n_batch_size,n_time_steps,n_features)
    def get_test_data(self, time_step_count=0):
        return self._get_data(self.test_items,time_step_count)

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
        return self._get_first_char_one_hot_labels(self.next_batch_items)

    # (n_batch_size,n_unique_chars)
    def get_test_first_char_one_hot_labels(self):
        return self._get_first_char_one_hot_labels(self.test_items)

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
        return self._get_labels(self.next_batch_items)

    def get_test_labels(self):
        return self._get_labels(self.test_items)

    def _get_labels(self,items):
        labels = []

        for item in items:
            labels.append(item.get_label())
        return labels

    def get_unique_chars(self):
        chars = []
        for item in self.all_items:
            label = item.get_label()
            for char in label:
                if not char in chars:
                    chars.append(char)
        return sorted(chars)

    def get_max_image_width(self):
        return self.get_max_time_steps()

    def get_max_time_steps(self):
        max_time_steps = 0
        for item in self.all_items:
            if (item.get_time_step_count() > max_time_steps):
                max_time_steps = item.get_time_step_count()
        return max_time_steps