# Outputs page images with word boxes as rectangles

from toolbox import wordio2
from PIL import Image
from PIL import ImageDraw
import os
import shutil
import dirs

pages_dir_path = dirs.STANFORD_PAGES_DIR_PATH
pages_with_boxes_dir_path = dirs.BASE_DIR_PATH + "/pages_with_word_boxes/"

if os.path.exists(pages_with_boxes_dir_path):
    shutil.rmtree(pages_with_boxes_dir_path)
os.makedirs(pages_with_boxes_dir_path)

for page_image_name in [f for f in os.listdir(pages_dir_path) if f.endswith(".jpg")]:
    print page_image_name
    page_image_file_path = os.path.join(pages_dir_path,page_image_name)
    words_file_path = os.path.join(dirs.WORDS_DIR_PATH,page_image_name.replace(".jpg",".words"))

    page_image = Image.open(page_image_file_path)
    page_draw = ImageDraw.Draw(page_image)

    lines, image_name = wordio2.read(words_file_path)

    for line in lines:
        print line
        for lw in range(5):
            box = (line.left+lw,line.top+lw,line.right-lw,line.bottom-lw)
            page_draw.rectangle(box,outline="green")

        words = line.words
        for word in words:
            for lw in range(5):
                box = (word.left+lw,word.top+lw,word.right-lw,word.bottom-lw)
                page_draw.rectangle(box,outline="blue")

    page_image.save(os.path.join(pages_with_boxes_dir_path,page_image_name))