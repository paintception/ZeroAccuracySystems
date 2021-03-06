import dirs
import os
import shutil
from PIL import Image

source_dir_path = dirs.KNMP_CHAR_BOXES_DIR_PATH
target_dir_path = dirs.KNMP_PROCESSED_CHAR_BOXES_DIR_PATH

# Delete target directory
if os.path.exists(target_dir_path):
    shutil.rmtree(target_dir_path)
os.makedirs(target_dir_path)

max_width = 0
for source_label_dir_name in [f for f in os.listdir(source_dir_path) if os.path.isdir(os.path.join(source_dir_path, f))]:
    source_label_dir_path = os.path.join(source_dir_path, source_label_dir_name)
    target_label_dir_path = os.path.join(target_dir_path, source_label_dir_name)

    os.makedirs(target_label_dir_path)

    source_file_names = [f for f in os.listdir(source_label_dir_path) if f.endswith(".png")]
    for source_file_name in source_file_names:
        source_file_path = os.path.join(source_label_dir_path, source_file_name)
        target_file_path = os.path.join(target_label_dir_path, source_file_name)

        print(target_file_path)

        image = Image.open(source_file_path)
        image = image.convert("LA") # Greyscale

        #print("%d\t%d" % (image.width, image.height))

        # Char width: average 29.9, std 8.8
        # Char height: average 60.0, std 11.4

        resize_ratio = 0.5
        new_width = int(image.width*resize_ratio)
        if (new_width > max_width):
            max_width = new_width
        new_height = 28
        image = image.resize((new_width,new_height))

        image.save(target_file_path)

        #shutil.copyfile(source_file_path,target_file_path)
print("Max width:",max_width)