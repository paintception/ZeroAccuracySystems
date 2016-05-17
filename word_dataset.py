import prepare_features as pf
import os
import random
import copy

class WordDataItem(object):
    def __init__(self,file_path):
        self.file_path = file_path
        self.label = pf.get_word_label_from_filename(self.file_path)
        self.load_data()

    def load_data(self):
        self.data = pf.get_feature_data(self.file_path)

    def get_data(self):
        return self.data

    def get_data_with_fixed_time_step_count(self,time_step_count):
        tmp_data = copy.copy(self.data) # Weak copy timesteps
        if (len(tmp_data) > time_step_count):
            tmp_data = tmp_data[:time_step_count] # Cut off some timesteps
        else:
            # Add timesteps
            for i in range(self.get_time_step_count(), time_step_count):
                tmp_data.append([0] * self.get_feature_count())
        return tmp_data

    def get_label(self):
        return self.label

    def get_width(self):
        return len(self.data)

    def get_height(self):
        return len(self.data[0])

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
            if (word_data_item.get_width() <= self.max_image_width):
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

    def init_train_batch(self):
        self.train_items_for_batch = []
        self.next_batch_items = []

    def prepare_next_train_batch(self, batch_size):
        self.next_train_batch_indexes = []
        for b in range(batch_size):
            if len(self.train_items_for_batch) == 0:
                self.train_items_for_batch = copy.copy(self.train_items) # Copy only references
                random.shuffle(self.train_items_for_batch)
            train_item = self.train_items_for_batch.pop()
            self.next_batch_items.append(train_item)

    def get_train_batch_data(self,time_step_count=0):
        batch_data = []

        for item in self.next_batch_items:
            if time_step_count == 0:
                batch_data.append(item.get_data())
            else:
                batch_data.append(item.get_data_with_fixed_time_step_count(time_step_count))
        return batch_data

    def get_train_batch_label_lengths(self):
        batch_label_lengths = []

        for item in self.next_batch_items:
            batch_label_lengths.append(len(item.label))

        return batch_label_lengths

    def get_unique_chars(self):
        chars = []
        for item in self.all_items:
            label = item.label
            for char in label:
                if not char in chars:
                    chars.append(char)
        return sorted(chars)