import dirs
import os
import shutil
from PIL import Image
import random
import prepare_features as pf

def sheer_image(image,sheer_factor=0):
    if sheer_factor == 0:
        return image
    width, height = image.size
    m = sheer_factor
    xshift = abs(m) * width
    new_width = width + int(round(xshift))
    return image.transform((new_width, height), Image.AFFINE,
                        (1, m, -xshift if m > 0 else 0, 0, 1, 0), Image.BICUBIC)

# source_dir_path = dirs.STANFORD_WORD_BOXES_DIR_PATH
# target_dir_path = dirs.STANFORD_PROCESSED_WORD_BOXES_DIR_PATH
#source_dir_path = dirs.KNMP_WORD_BOXES_DIR_PATH
# target_dir_path = dirs.KNMP_PROCESSED_WORD_BOXES_DIR_PATH
source_dir_path = "/Users/rmencis/Dropbox/Studies/RUG/Handwriting_recognition/HWR_Share/word_boxes/Otsu_Binarization_Stanford"
#target_dir_path = dirs.KNMP_PROCESSED_WORD_BOXES_DIR_PATH
target_dir_path = "/Users/rmencis/RUG/Handwriting_Recognition/word_boxes_processed_otsu_10x_expanded/Stanford"

train_dir_path = os.path.join(target_dir_path,"train")
test_dir_path = os.path.join(target_dir_path,"test")

test_file_count = 100

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
train_file_counter = 0
for source_file_name in source_file_names:
    source_file_counter += 1
    train = False
    test = True
    if (source_file_counter > test_file_count):
        train = True
        test = False

    source_file_path = os.path.join(source_dir_path, source_file_name)

    print(source_file_path)

    image = Image.open(source_file_path)
    # image = image.convert("LA") # Greyscale

    if train:
        train_images = [image]
        for i in range(10):
            train_image = image

            # Crop vertically
            crop_ratio = 0.1
            new_top = int(train_image.height * random.uniform(-crop_ratio, crop_ratio))
            new_bottom = train_image.height + int(train_image.height * random.uniform(-crop_ratio, crop_ratio))
            crop_box = (0, new_top, train_image.width, new_bottom)
            train_image = train_image.crop(crop_box)

            # Sheer
            sheer_ratio = random.uniform(-0.20,0.20)
            train_image = sheer_image(train_image, (sheer_ratio))
            train_images.append(train_image)

        # Save training images
        for train_image in train_images:
            train_image = pf.preprocess_image(train_image)
            train_file_counter += 1
            target_file_name = str(train_file_counter).rjust(6, "0") + "_" + source_file_name
            target_file_path = os.path.join(train_dir_path, target_file_name)
            train_image.save(target_file_path)

    if test:
        image = pf.preprocess_image(image)
        target_file_path = os.path.join(test_dir_path, source_file_name)
        image.save(target_file_path)

# print("Average width:",int(total_width / len(source_file_names)))
# print("Max width:",max_width)