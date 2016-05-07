# Cuts out all word boxes and saves as separate files

from toolbox import wordio
from PIL import Image
import os
import shutil
import dirs
import sys

# Write word or character box to file
def write_box(page_image, box_dir_path, box, label_text):
    try:
        box_image = page_image.crop(box)
        box_image_name = page_image_name.replace(".jpg", "_") + str(box[0]) + "_" + str(box[1])  + "_" + str(box[2]) + "_" + str(box[3])+ "_" + label_text + ".png"

        label_dir_name = label_text
        if label_dir_name == ".":
            label_dir_name = "_."
        if len(label_dir_name) == 0:
            label_dir_name = "_"
        label_dir_path = os.path.join(box_dir_path,label_dir_name)
        if not os.path.exists(label_dir_path):
            os.makedirs(label_dir_path)
        box_image.save(os.path.join(label_dir_path, box_image_name))
    except:
        print("Unexpected error:", sys.exc_info()[0])

# Parameters
pages_dir_path = os.path.join(dirs.BASE_DIR_PATH,"relabeled_pages")
word_image_dir_path = dirs.KNMP_WORD_BOXES_DIR_PATH
char_image_dir_path = dirs.KNMP_CHAR_BOXES_DIR_PATH

# Delete and create directories
if os.path.exists(word_image_dir_path):
    shutil.rmtree(word_image_dir_path)
os.makedirs(word_image_dir_path)

if os.path.exists(char_image_dir_path):
    shutil.rmtree(char_image_dir_path)
os.makedirs(char_image_dir_path)

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
            write_box(page_image,word_image_dir_path,box,word.text)

            chars = word.characters
            for char in chars:
                box = (char.left,char.top,char.right,char.bottom)
                write_box(page_image,char_image_dir_path,box,char.text)

