import dirs
import os
import shutil
import cv2

def preprocess_image(image):
    # Otsu's thresholding after Gaussian filtering
    blured_image = cv2.GaussianBlur(image,(5,5),0)
    histogram,binary_image = cv2.threshold(blured_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return binary_image

source_dir_path = dirs.BASE_DIR_PATH + "/knmp_word_boxes/"
target_dir_path = dirs.BASE_DIR_PATH + "/knmp_word_boxes_processed/"

# Delete target directory
if os.path.exists(target_dir_path):
    shutil.rmtree(target_dir_path)
os.makedirs(target_dir_path)

for source_label_dir_name in [f for f in os.listdir(source_dir_path) if os.path.isdir(os.path.join(source_dir_path, f))]:
    source_label_dir_path = os.path.join(source_dir_path, source_label_dir_name)
    target_label_dir_path = os.path.join(target_dir_path, source_label_dir_name)

    os.makedirs(target_label_dir_path)

    source_file_names = [f for f in os.listdir(source_label_dir_path) if f.endswith(".png")]
    for source_file_name in source_file_names:
        source_file_path = os.path.join(source_label_dir_path, source_file_name)
        target_file_path = os.path.join(target_label_dir_path, source_file_name)
        shutil.copyfile(source_file_path,target_file_path)