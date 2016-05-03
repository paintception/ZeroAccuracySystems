# Cuts out all word boxes and saves as separate files

from toolbox import wordio
from PIL import Image
import os
import shutil
import dirs
import sys

pages_dir_path = dirs.STANFORD_PAGES_DIR_PATH
char_image_dir_path = dirs.BASE_DIR_PATH + "/char_boxes/"

if os.path.exists(char_image_dir_path):
    shutil.rmtree(char_image_dir_path)
os.makedirs(char_image_dir_path)

for page_image_name in [f for f in os.listdir(pages_dir_path) if f.endswith(".jpg")]:
    print page_image_name
    page_image_file_path = os.path.join(pages_dir_path,page_image_name)
    words_file_path = os.path.join(dirs.WORDS_DIR_PATH,page_image_name.replace(".jpg",".words"))

    page_image = Image.open(page_image_file_path)

    try:
        lines, image_name = wordio.read(words_file_path)

        for line in lines:
            words = line
            for word in words:
                chars = word.characters
                for char in chars:
                    box = (char.left,char.top,char.right,char.bottom)
                    try:
                        char_image = page_image.crop(box)
                        char_image_name = page_image_name.replace(".jpg", "_") + str(char.left) + "_" + str(char.top) + "_" + char.text + ".png"

                        text_dir_path = os.path.join(char_image_dir_path,char.text)
                        if not os.path.exists(text_dir_path):
                            os.makedirs(text_dir_path)
                        char_image.save(os.path.join(text_dir_path, char_image_name))
                    except:
                        print "Unexpected error:", sys.exc_info()[0]
    except:
        print "Unexpected error:", sys.exc_info()[0]
