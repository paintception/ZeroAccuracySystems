# Author - Roberts
# Cuts out all character/word boxes and saves as separate files in directories named as labels

from toolbox import wordio
from PIL import Image
import os
import shutil
import dirs
import sys

# Parameters
pages_dir_path = os.path.join(dirs.BASE_DIR_PATH,"relabeled_pages")
#pages_dir_path = dirs.STANFORD_PAGES_DIR_PATH
word_image_dir_path = dirs.KNMP_WORD_BOXES_DIR_PATH

# Delete and create directories
if os.path.exists(word_image_dir_path):
    shutil.rmtree(word_image_dir_path)
os.makedirs(word_image_dir_path)

word_counter = 0
char_counter = 0
# Go through page files
for page_image_name in [f for f in os.listdir(pages_dir_path) if f.endswith(".jpg")]:
    print(page_image_name)
    page_image_file_path = os.path.join(pages_dir_path,page_image_name)

    # Find label file
    words_file_path = os.path.join(dirs.LABELS_DIR_PATH, page_image_name.replace(".jpg", ".words"))
    if not os.path.exists(words_file_path):
        continue

    # Open page image
    page_image = Image.open(page_image_file_path)

    # Read label XML
    lines, image_name = wordio.read(words_file_path)

    for line in lines:
        words = line
        for word in words:
            box = (word.left,word.top,word.right,word.bottom)
            try:
                box_image = page_image.crop(box)
                box_image_name = page_image_name.replace(".jpg", "_") + str(box[0]) + "_" + str(box[1]) + "_" + str(
                    box[2]) + "_" + str(box[3]) + "_" + word.text + ".png"

                box_image.save(os.path.join(word_image_dir_path, box_image_name))
                word_counter += 1
                char_counter = char_counter + len(word.text)
            except:
                print("Unexpected error:", sys.exc_info()[0])

print("Words:",word_counter)
print("Characters:",char_counter)