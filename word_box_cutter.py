# Cuts out all word boxes and saves as separate files

from toolbox import wordio
from PIL import Image
import os
import shutil
import dirs

pages_dir_path = dirs.STANFORD_PAGES_DIR_PATH
word_image_dir_path = dirs.BASE_DIR_PATH + "/word_boxes/"

if os.path.exists(word_image_dir_path):
    shutil.rmtree(word_image_dir_path)
os.makedirs(word_image_dir_path)

for page_image_name in [f for f in os.listdir(pages_dir_path) if f.endswith(".jpg")]:
    print page_image_name
    page_image_file_path = os.path.join(pages_dir_path,page_image_name)
    words_file_path = os.path.join(dirs.WORDS_DIR_PATH,page_image_name.replace(".jpg",".words"))

    page_image = Image.open(page_image_file_path)

    lines, image_name = wordio.read(words_file_path)

    for line in lines:
        words = line
        for word in words:
            box = (word.left,word.top,word.right,word.bottom)
            word_image = page_image.crop(box)
            word_image_name = page_image_name.replace(".jpg", "_") + str(word.left) + "_" + str(word.top) + "_" + word.text + ".png"

            word_image.save(os.path.join(word_image_dir_path, word_image_name))

            text_dir_path = os.path.join(word_image_dir_path,word.text)
            if not os.path.exists(text_dir_path):
                os.makedirs(text_dir_path)
            word_image.save(os.path.join(text_dir_path, word_image_name))

            #print word
            # chars = word.characters
            # for char in chars:
            #     print char

