import dirs
import os
import shutil
from PIL import Image

dir_path = dirs.STANFORD_PROCESSED_WORD_BOXES_DIR_PATH + "/train"

file_names = [f for f in os.listdir(dir_path) if f.endswith(".png")]

total_width = 0
total_height = 0
max_width = 0
max_height = 0
for file_name in file_names:
    file_path = os.path.join(dir_path,file_name)
    print(file_path)

    image = Image.open(file_path)

    total_width = total_width + image.width
    total_height = total_height + image.height

    if image.width > max_width:
        max_width = image.width

    if image.height > max_height:
        max_height = image.height

avg_width = total_width / len(file_names)
avg_height = total_height / len(file_names)

print("Average width:",avg_width,"Max. width:",max_width)
print("Average height:",avg_height,"Max. height:",max_height)