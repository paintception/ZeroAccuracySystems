import dirs
import os
import shutil

source_dir_path = dirs.KNMP_ADDITIONAL_WORD_BOXES_DIR_PATH
target_dir_path = dirs.KNMP_WORD_BOXES_DIR_PATH

for word_image_name in [f for f in os.listdir(source_dir_path) if f.endswith(".png")]:
    word_image_file_path = os.path.join(source_dir_path,word_image_name)

    label = word_image_name.replace(".png","")
    last_sep = label.rfind("_")
    label = label[last_sep+1:]

    label_dir_path = os.path.join(target_dir_path,label)
    if not os.path.exists(label_dir_path):
        os.makedirs(label_dir_path)

    label_file_path = os.path.join(label_dir_path,word_image_name)
    shutil.copyfile(word_image_file_path, label_file_path)

    print(label)
