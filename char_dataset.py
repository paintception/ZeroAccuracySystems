import prepare_features as pf
import os
import random

# The class, which keeps dataset (labels, image data etc.) and provides training/test data
class CharDataSet(object):
    def __init__(self, dir_path, train_ratio=0.9):
        self.dir_path = dir_path
        self.train_ratio = train_ratio

        self.load_classes()
        self.load_data()
        self.prepare_training_and_test_data()

        self.train_indexes_for_batch = []
        self.next_batch_indexes = []

    def load_classes(self):
        self.class_names = pf.get_classes(self.dir_path)

    def load_data(self):
        self.data = []
        self.file_paths = []
        self.label_names = []
        self.label_indexes = []
        self.one_hot_labels = []
        for class_name in self.class_names:
            class_dir_path = os.path.join(self.dir_path,class_name)
            class_index = self.class_names.index(class_name)
            class_one_hot = pf.get_one_hot(class_index,len(self.class_names)) # [0,0,0,1,0,0,0,0,0,0] (class 3 from 10 classes)

            image_file_names = [f for f in os.listdir(class_dir_path) if f.endswith(".png")]

            for image_file_name in image_file_names:
                image_file_path = os.path.join(class_dir_path,image_file_name)

                time_steps_with_features = pf.get_feature_data(image_file_path, 37)
                self.data.append(time_steps_with_features)
                self.file_paths.append(image_file_path)
                self.label_names.append(class_name)
                self.label_indexes.append(class_index)
                self.one_hot_labels.append(class_one_hot)

                if len(self.data) % 100 == 0:
                    print("Loaded %d images" % len(self.data));
        print("Loaded all %d images" % len(self.data));

    def prepare_training_and_test_data(self):
        all_indexes = list(range(len(self.data)))
        random.shuffle(all_indexes)
        training_count = int(len(self.data) * self.train_ratio)

        self.train_indexes = []
        for i in range(training_count):
            self.train_indexes.append(all_indexes.pop())

        self.test_indexes = all_indexes

        # Cache test data
        self.test_data = []
        self.test_one_hot_labels = []
        for index in self.test_indexes:
            self.test_data.append(self.data[index])
            self.test_one_hot_labels.append(self.one_hot_labels[index])

    def prepare_next_batch(self, batch_size):
        self.next_batch_indexes = []
        for b in range(batch_size):
            if len(self.train_indexes_for_batch) == 0:
                self.train_indexes_for_batch = list(self.train_indexes)
                random.shuffle(self.train_indexes_for_batch)
            train_index = self.train_indexes_for_batch.pop()
            self.next_batch_indexes.append(train_index)

    def get_batch_data(self):
        batch_data = []

        for index in self.next_batch_indexes:
            data = self.data[index]
            batch_data.append(data)
        return batch_data

    def get_batch_one_hot_labels(self):
        batch_one_hot_labels = []

        for index in self.next_batch_indexes:
            one_hot_label = self.one_hot_labels[index]
            batch_one_hot_labels.append(one_hot_label)
        return batch_one_hot_labels

    def get_test_data(self):
        return self.test_data

    def get_test_one_hot_labels(self):
        return self.test_one_hot_labels

    def get_total_item_count(self):
        return len(self.data)

    def get_train_item_count(self):
        return len(self.train_indexes)

    def get_test_item_count(self):
        return len(self.test_indexes)

    def get_time_step_count(self):
        return len(self.data[0])

    def get_feature_count(self):
        return len(self.data[0][0])

    def get_class_count(self):
        return len(self.class_names)