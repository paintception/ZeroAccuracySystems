# Author - Roberts, Yaroslav
# Cuts out all character/word boxes and saves as separate files in directories named as labels

from toolbox import wordio
from PIL import Image
import os
import shutil
import dirs
import pickle
import tqdm

if __name__ == "__main__":
    # Parameters

    # pages_dir_path = dirs.KNMP_PAGES_DIR_PATH
    # word_image_dir_path = dirs.KNMP_WORD_BOXES_DIR_PATH
    pages_dir_path = dirs.STANFORD_PAGES_DIR_PATH
    word_image_dir_path = dirs.STANFORD_WORD_BOXES_DIR_PATH

    # Delete and create directories
    if os.path.exists(word_image_dir_path):
        shutil.rmtree(word_image_dir_path)
    os.makedirs(word_image_dir_path)

    word_counter = 0
    char_counter = 0
    # Go through page files
    word_info = []

    image_files = [f for f in os.listdir(pages_dir_path) if f.endswith(".jpg")]

    pbar = tqdm.tqdm(desc="Cutting pages", total=len(image_files))

    for page_image_name in image_files:

        page_image_file_path = os.path.join(pages_dir_path, page_image_name)

        # Find label file
        words_file_path = os.path.join(dirs.LABELS_DIR_PATH, page_image_name.replace(".jpg", ".words"))
        if not os.path.exists(words_file_path):
            continue

        # Open page image
        page_image = Image.open(page_image_file_path)

        # Read label XML
        lines, image_name = wordio.read(words_file_path)

        for line in lines:
            for word in line:
                box = (word.left, word.top, word.right, word.bottom)
                try:
                    box_image = page_image.crop(box)
                    box_image_name = page_image_name.replace(".jpg", "_") + str(box[0]) + "_" + str(box[1]) + "_" + str(
                        box[2]) + "_" + str(box[3]) + "_" + word.text + ".png"

                    box_image.save(os.path.join(word_image_dir_path, box_image_name))
                    word_counter += 1

                    word_length = word.right - word.left
                    char_labels = []
                    char_positions = []
                    for character in word.characters:
                        char_labels.append(character.text)
                        char_position = (character.left - word.left) + int((character.right - character.left) / 2)
                        char_positions.append(char_position)
                        char_counter += 1

                    if char_labels:
                        word_info.append({
                            'box_image_name': box_image_name,
                            'word_length': word_length,
                            'char_labels': char_labels,
                            'char_positions': char_positions
                        })

                except Exception as e:
                    raise e
                    # print("Unexpected error:", sys.exc_info()[0])

        pbar.update(1)

    pbar.close()

    with open(os.path.join(word_image_dir_path, 'word_info'), 'wb') as word_info_file:
        pickle.dump(word_info, word_info_file)

    print("Words:", word_counter)
    print("Characters:", char_counter)
