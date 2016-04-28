from toolbox import wordio
from PIL import Image
import os
import shutil

words_dir_path = "/Users/rmencis/RUG/Handwriting_Recognition/charannotations/"
page_image_dir_path = "/Users/rmencis/RUG/Handwriting_Recognition/pages/Stanford"
word_image_dir = "/Users/rmencis/RUG/Handwriting_Recognition/word_images/"

if os.path.exists(word_image_dir):
    shutil.rmtree(word_image_dir)
os.makedirs(word_image_dir)

for page_image_name in [f for f in os.listdir(page_image_dir_path) if f.endswith(".jpg")]:
    print page_image_name
    page_image_file_path = os.path.join(page_image_dir_path,page_image_name)
    words_file_path = os.path.join(words_dir_path,page_image_name.replace(".jpg",".words"))

    page_image = Image.open(page_image_file_path)

    lines, image_name = wordio.read(words_file_path)

    for line in lines:
        words = line
        for word in words:
            box = (word.left,word.top,word.right,word.bottom)
            word_image = page_image.crop(box)
            word_image.save(os.path.join(word_image_dir,page_image_name.replace(".jpg","_") + str(word.left) + "_" + str(word.top) + ".png"))
            #print word
            # chars = word.characters
            # for char in chars:
            #     print char

