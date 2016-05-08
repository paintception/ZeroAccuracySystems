import prepare_features as pf
import os
import random

# The class, which keeps dataset (labels, image data etc.)
class DataSet(object):
    def __init__(self, dir_path, train_ratio=0.9):
        self.dir_path = dir_path
        self.train_ratio = train_ratio

        self.load_classes()
        self.load_data()
        self.prepare_training_and_test_data()

        self.batch_indexes = []

    def load_classes(self):
        self.class_names = pf.get_classes(self.dir_path)

    def load_data(self):
        self.data = []
        self.file_paths = []
        self.label_names = []
        self.label_indexes = []
        for class_name in self.class_names:
            class_dir_path = os.path.join(self.dir_path,class_name)
            class_index = self.class_names.index(class_name)

            image_file_names = [f for f in os.listdir(class_dir_path) if f.endswith(".png")]

            for image_file_name in image_file_names:
                image_file_path = os.path.join(class_dir_path,image_file_name)

                time_steps_with_features = pf.get_image_time_steps_with_features(image_file_path)
                self.data.append(time_steps_with_features)
                self.file_paths.append(image_file_path)
                self.label_names.append(class_name)
                self.label_indexes.append(class_index)

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

    def prepare_next_batch(self):
        if len(self.batch_indexes) == 0:
            self.batch_indexes = list(self.train_indexes)

    def get_all_item_count(self):
        return len(self.data)

    def get_train_item_count(self):
        return len(self.train_indexes)

    def get_test_item_count(self):
        return len(self.test_indexes)