import datetime
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
from word_dataset import WordDataSet,WordDataItem
import dirs
import random

# Read data set

dataset = WordDataSet(dirs.KNMP_PROCESSED_WORD_BOXES_DIR_PATH)

print("Total items:",dataset.get_total_item_count())
print("Training items:",dataset.get_train_item_count())
print("Test items:",dataset.get_test_item_count())

