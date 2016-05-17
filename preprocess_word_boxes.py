import dirs
import os
import shutil
from PIL import Image
import random

source_dir_path = dirs.KNMP_WORD_BOXES_DIR_PATH
target_dir_path = dirs.KNMP_PROCESSED_WORD_BOXES_DIR_PATH

train_dir_path = os.path.join(target_dir_path,"train")
test_dir_path = os.path.join(target_dir_path,"test")

train_ratio = 0.9

# Delete and create target directories
if os.path.exists(target_dir_path):
    shutil.rmtree(target_dir_path)

os.makedirs(train_dir_path)
os.makedirs(test_dir_path)

max_width = 0
source_file_names = [f for f in os.listdir(source_dir_path) if f.endswith(".png")]
random.shuffle(source_file_names)

file_counter = 0
train_file_count = int(len(source_file_names) * train_ratio)
for source_file_name in source_file_names:
    file_counter += 1
    train = True
    test = False
    if (file_counter > train_file_count):
        train = False
        test = True

    source_file_path = os.path.join(source_dir_path, source_file_name)
    target_file_path = os.path.join(train_dir_path, source_file_name)
    if test:
        target_file_path = os.path.join(test_dir_path, source_file_name)

    print(target_file_path)

    image = Image.open(source_file_path)
    image = image.convert("LA") # Greyscale

    # Word width: average 91.4, std 65.6
    # Word height: average 56.7, std 11.9

    resize_ratio = 0.5
    new_width = int(image.width*resize_ratio)
    if (new_width > max_width):
        max_width = new_width
    new_height = 28
    image = image.resize((new_width,new_height))

    image.save(target_file_path)
print("Max width:",max_width)