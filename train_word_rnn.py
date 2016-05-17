import datetime
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
from word_dataset import WordDataSet,WordDataItem
import dirs
import random

# Read data set

dataset = WordDataSet(dirs.KNMP_PROCESSED_WORD_BOXES_DIR_PATH,max_image_width=100)

print("Total items:",dataset.get_total_item_count())
print("Training items:",dataset.get_train_item_count())
print("Test items:",dataset.get_test_item_count())

dataset.prepare_next_train_batch(batch_size=100)
data = dataset.get_train_batch_data(time_step_count=150)
for d in data:
    print(len(d))