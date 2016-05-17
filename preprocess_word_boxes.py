import dirs
import os
import shutil
from PIL import Image
import random

def sheer_image(image,sheer_factor=0):
    width, height = image.size
    m = sheer_factor
    xshift = abs(m) * width
    new_width = width + int(round(xshift))
    return image.transform((new_width, height), Image.AFFINE,
                        (1, m, -xshift if m > 0 else 0, 0, 1, 0), Image.BICUBIC)

def resize_image(image):
    resize_ratio = 0.5
    new_width = int(image.width * resize_ratio)
    new_height = 28
    return image.resize((new_width, new_height))


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

source_file_names = [f for f in os.listdir(source_dir_path) if f.endswith(".png")]
random.shuffle(source_file_names)

source_file_counter = 0
max_width = 0
total_width = 0
train_file_count = int(len(source_file_names) * train_ratio)
train_file_counter = 0
for source_file_name in source_file_names:
    source_file_counter += 1
    train = True
    test = False
    if (source_file_counter > train_file_count):
        train = False
        test = True

    source_file_path = os.path.join(source_dir_path, source_file_name)

    print(source_file_path)

    image = Image.open(source_file_path)
    image = image.convert("LA") # Greyscale

    if train:
        for sheer_ratio in range(-2,3,1):
            train_image = image
            if (sheer_ratio != 0):
                train_image = sheer_image(train_image,sheer_ratio*0.1)
            train_image = resize_image(train_image)
            train_file_counter += 1
            target_file_name = str(train_file_counter).rjust(6, "0") + "_" + source_file_name
            target_file_path = os.path.join(train_dir_path, target_file_name)
            train_image.save(target_file_path)

    if test:
        image = resize_image(image)
        target_file_path = os.path.join(test_dir_path, source_file_name)
        image.save(target_file_path)

# print("Average width:",int(total_width / len(source_file_names)))
# print("Max width:",max_width)