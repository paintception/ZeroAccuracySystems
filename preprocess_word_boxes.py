import dirs
import os
import shutil
from PIL import Image

source_dir_path = dirs.KNMP_WORD_BOXES_DIR_PATH
target_dir_path = dirs.KNMP_PROCESSED_WORD_BOXES_DIR_PATH

# Delete target directory
if os.path.exists(target_dir_path):
    shutil.rmtree(target_dir_path)
os.makedirs(target_dir_path)

max_width = 0
source_file_names = [f for f in os.listdir(source_dir_path) if f.endswith(".png")]
for source_file_name in source_file_names:
    source_file_path = os.path.join(source_dir_path, source_file_name)
    target_file_path = os.path.join(target_dir_path, source_file_name)

    print(target_file_path)

    image = Image.open(source_file_path)
    image = image.convert("LA") # Greyscale

    # Word width: average 91.4, std 65.6
    # Word height: average 56.7, std 11.9

    resize_ratio = 0.5
    new_width = int(image.width*resize_ratio)
    if (new_width > max_width):
        max_width = new_width
    new_height = 28
    image = image.resize((new_width,new_height))

    image.save(target_file_path)
print("Max width:",max_width)