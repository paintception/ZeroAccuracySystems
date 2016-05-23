import prepare_features as pf
import os
import random
import copy
import pickle
import tqdm


class WordDataItem(object):
    def __init__(self, file_path, label, label_timesteps):
        self.file_path = file_path
        self.label = label
        self.label_timesteps = label_timesteps
        self.data = pf.get_feature_data(self.file_path)

    def get_data_with_fixed_time_step_count(self, time_step_count):
        tmp_data = copy.copy(self.data)  # Weak copy timesteps
        if (len(tmp_data) > time_step_count):
            tmp_data = tmp_data[:time_step_count]  # Cut off some timesteps
        else:
            # Add timesteps
            for i in range(self.get_time_step_count(), time_step_count):
                tmp_data.append([0] * self.get_feature_count())
        return tmp_data

    def get_data(self):
        return self.data

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
# noinspection PyMethodMayBeStatic
class WordDataSet(object):
    def __init__(self, dir_path, max_image_width=1000):
        self.dir_path = dir_path
        self.max_image_width = max_image_width

        self.load_data()
        self.init_train_batch()

    def load_data(self):
        self.train_items = self.load_data_items("train")
        self.test_items = self.load_data_items("test")
        self.all_items = self.train_items + self.test_items
        self.unique_chars = self.get_unique_chars()

    def load_data_items(self, train_vs_test):
        items = []
        file_dir_path = os.path.join(self.dir_path, train_vs_test)

        with open(os.path.join(file_dir_path, 'word_info'), 'rb') as word_info_file:
            word_info = pickle.load(word_info_file)

        pbar = tqdm.tqdm(desc="Loading " + train_vs_test, total=len(word_info))
        for word in word_info:
            for word_file in word['ready_files']:

                word_data_item = WordDataItem(word_file, word['char_labels'], word['char_positions'])
                if (word_data_item.get_width() <= self.max_image_width):
                    items.append(word_data_item)

            pbar.update(1)

        pbar.close()
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
                self.train_items_for_batch = copy.copy(self.train_items)  # Copy only references
                random.shuffle(self.train_items_for_batch)
            train_item = self.train_items_for_batch.pop()
            self.next_batch_items.append(train_item)

    # (n_batch_size,n_time_steps,n_features)
    def get_train_batch_data(self, time_step_count=0):
        return self._get_data(self.next_batch_items, time_step_count)

    # (n_batch_size,n_time_steps,n_features)
    def get_test_data(self, time_step_count=0):
        return self._get_data(self.test_items, time_step_count)

    def _get_data(self, items, time_step_count=0):
        data = []

        for item in items:
            if time_step_count == 0:
                data.append(item.get_data())
            else:
                data.append(item.get_data_with_fixed_time_step_count(time_step_count))
        return data

    def get_train_batch_sequence_lengths(self, time_step_count=None):
        return self._get_sequence_length(self.next_batch_items, time_step_count)

    def get_test_sequence_lengths(self, time_step_count=None):
        return self._get_sequence_length(self.test_items, time_step_count)

    def _get_sequence_length(self, items, time_step_count=None):
        if time_step_count:
            return [item.get_time_step_count() if item.get_time_step_count() < time_step_count else time_step_count
                    for item in items]
        else:
            return [item.get_time_step_count() for item in items]

    def get_train_batch_labels_with_timesteps(self, time_step_count=None):
        return self._get_labels_with_timesteps(self.next_batch_items, time_step_count)

    def get_test_labels_with_timesteps(self, time_step_count=None):
        return self._get_labels_with_timesteps(self.test_items, time_step_count)

    def _get_labels_with_timesteps(self, items, time_step_count=None):
        label_index = []
        labels = []

        for idx, item in enumerate(items):
            if time_step_count:
                label_index_item = [[idx, x] for x in item.label_timesteps if x < time_step_count]
            else:
                label_index_item = [[idx, x] for x in item.label_timesteps]
            labels += map(lambda x: self.unique_chars.index(x), item.label[0:len(label_index_item)])
            label_index += label_index_item

        return label_index, labels

    # (n_batch_size,n_unique_chars)
    def get_train_batch_first_char_one_hot_labels(self):
        return self._get_first_char_one_hot_labels(self.next_batch_items)

    # (n_batch_size,n_unique_chars)
    def get_test_first_char_one_hot_labels(self):
        return self._get_first_char_one_hot_labels(self.test_items)

    def _get_first_char_one_hot_labels(self, items):
        one_hot_labels = []
        unique_chars = self.get_unique_chars()

        for item in items:
            label = item.get_label()
            char = label[0]
            char_index = unique_chars.index(char)
            one_hot_labels.append(pf.get_one_hot(char_index, len(unique_chars)))
        return one_hot_labels

    def get_train_batch_labels(self):
        return self._get_labels(self.next_batch_items)

    def get_test_labels(self):
        return self._get_labels(self.test_items)

    def _get_labels(self, items):
        labels = []

        for item in items:
            labels.append(item.get_label())
        return labels

    # def get_train_batch_label_lengths(self):
    #     batch_label_lengths = []
    #
    #     for item in self.next_batch_items:
    #         batch_label_lengths.append(len(item.label))
    #
    #     return batch_label_lengths

    def get_unique_chars(self):
        chars = []
        for item in self.all_items:
            label = item.label
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


if __name__ == "__main__":
    import dirs
    wd = WordDataSet(dirs.STANFORD_PROCESSED_WORD_BOXES_DIR_PATH)
    pass
