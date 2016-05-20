# Author - Roberts

from toolbox import wordio
from PIL import Image
import os
import shutil
import dirs
import sys

# Parameters
labels_dir_path = dirs.LABELS_DIR_PATH
output_dir_path = "/Users/rmencis/RUG/Handwriting_Recognition/corrected_labels"

# Delete and create directories
if os.path.exists(output_dir_path):
    shutil.rmtree(output_dir_path)
os.makedirs(output_dir_path)

for label_file_name in [f for f in os.listdir(labels_dir_path) if f.endswith(".words")]:
    print(label_file_name)
    label_file_path = os.path.join(labels_dir_path,label_file_name)

    lines, image_name = wordio.read(label_file_path)

    for words in lines:
        for word in words:
            chars = word.characters
            text = ""
            for char in chars:
                text = text + char.text
            word.text = text

    output_file_path = os.path.join(output_dir_path,label_file_name)

    wordio.save(lines,output_file_path)

