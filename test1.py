from toolbox import wordio
from PIL import Image
import os

page_image_path = "/Users/rmencis/RUG/Handwriting_Recognition/pages/Stanford/Stanford-CCCC_0040.jpg"
words_path = "/Users/rmencis/RUG/Handwriting_Recognition/charannotations/Stanford-CCCC_0040.words"
word_image_dir = "/Users/rmencis/RUG/Handwriting_Recognition/tmp/"

page_image = Image.open(page_image_path)

lines, image_name = wordio.read(words_path)

for line in lines:
    words = line
    for word in words:
        box = (word.left,word.top,word.right,word.bottom)
        word_image = page_image.crop(box)
        word_image.save(os.path.join(word_image_dir,str(word.left) + "_" + str(word.top) + ".png"))
        print word
        chars = word.characters
        for char in chars:
            print char