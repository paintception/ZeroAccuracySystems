import dirs
import os
import shutil
from PIL import Image
import random
import pickle


def sheer_image(image, sheer_factor=0):
    if sheer_factor == 0:
        return image
    width, height = image.size
    m = sheer_factor
    xshift = abs(m) * width
    new_width = width + int(round(xshift))
    return image.transform((new_width, height), Image.AFFINE,
                           (1, m, -xshift if m > 0 else 0, 0, 1, 0), Image.BICUBIC)


def resize_image(image, resize_ratio=0.5):
    new_width = int(image.width * resize_ratio)
    new_height = 28
    return image.resize((new_width, new_height))


if __name__ == "__main__":
    source_dir_path = dirs.KNMP_WORD_BOXES_DIR_PATH
    target_dir_path = dirs.KNMP_PROCESSED_WORD_BOXES_DIR_PATH

    train_dir_path = os.path.join(target_dir_path, "train")
    test_dir_path = os.path.join(target_dir_path, "test")

    train_ratio = 0.9

    resize_ratio = 0.5

    # Delete and create target directories
    if os.path.exists(target_dir_path):
        shutil.rmtree(target_dir_path)

    os.makedirs(train_dir_path)
    os.makedirs(test_dir_path)

    with open(os.path.join(source_dir_path, 'word_info'), 'rb') as word_info_file:
        word_info = pickle.load(word_info_file)

    train_word_info = []
    test_word_info = []

    random.shuffle(word_info)

    source_file_counter = 0
    max_width = 0
    total_width = 0
    train_file_count = int(len(word_info) * train_ratio)
    train_file_counter = 0
    for word in word_info:
        source_file_counter += 1
        train = True
        test = False
        if source_file_counter > train_file_count:
            train = False
            test = True

        source_file_path = os.path.join(source_dir_path, word["box_image_name"])

        print(source_file_path)

        image = Image.open(source_file_path)
        image = image.convert("LA")  # Greyscale

        ready_files = []
        if train:
            for sheer_ratio in range(-2, 3):
                train_image = image
                train_image = sheer_image(train_image, (sheer_ratio * 0.05))
                train_image = resize_image(train_image, resize_ratio=resize_ratio)
                word['char_positions'] = [int(x * resize_ratio) for x in word['char_positions']]
                train_file_counter += 1
                target_file_name = str(train_file_counter).rjust(6, "0") + "_" + word["box_image_name"]
                target_file_path = os.path.join(train_dir_path, target_file_name)
                train_image.save(target_file_path)
                ready_files.append(target_file_path)

            word['ready_files'] = ready_files
            train_word_info.append(word)

        elif test:
            image = resize_image(image)
            target_file_path = os.path.join(test_dir_path, word["box_image_name"])
            image.save(target_file_path)
            ready_files.append(target_file_path)

            word['ready_files'] = ready_files
            test_word_info.append(word)

    with open(os.path.join(train_dir_path, 'word_info'), 'wb') as word_info_file:
        pickle.dump(train_word_info, word_info_file)
    with open(os.path.join(test_dir_path, 'word_info'), 'wb') as word_info_file:
        pickle.dump(test_word_info, word_info_file)

        # print("Average width:",int(total_width / len(word_info)))
        # print("Max width:",max_width)
