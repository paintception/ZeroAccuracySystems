from PIL import Image
from copy import copy, deepcopy
import os
import random

# Return 2D list (time_steps,features). Time steps = image width, Features = pixel values of 1px slices
def get_image_time_steps_with_features(image_file_path,min_time_steps=0):
    image = Image.open(image_file_path)

    # Get pixel values
    pixel_value_sum = 0
    time_steps_with_features = [] # 1px slices
    for x in range(image.width):
        time_step_features = [] # Pixel values of 1px slice (from top to down)
        for y in range(image.height):
            pixel = image.getpixel((x,y))
            pixel_value = pixel[0] / pixel[1] # Pixel value in range 0..1
            pixel_value_sum = pixel_value_sum + pixel_value
            time_step_features.append(pixel_value)
        time_steps_with_features.append(time_step_features)

    # Normalize around 0
    mean_pixel_value = pixel_value_sum / (image.width*image.height)
    for ts in range(len(time_steps_with_features)):
        time_step_features = time_steps_with_features[ts]
        for f in range(len(time_step_features)):
            time_step_features[f] = time_step_features[f] - mean_pixel_value

    # Pad with empty time steps
    if (min_time_steps > 0):
        for i in range(image.width, min_time_steps):
            # time_step_features = []
            # for j in range(image.height):
            #     time_step_features.append(random.random()-0.5)
            # time_steps_with_features.append(time_step_features)
            time_steps_with_features.append([0] * image.height)

    return time_steps_with_features

# Return class names
def get_classes(label_dir_path):
    class_names = sorted([f for f in os.listdir(label_dir_path) if os.path.isdir(os.path.join(label_dir_path, f)) and not f.startswith(".")])
    return class_names

def get_one_hot(index,length):
    one_hot = [0] * length
    one_hot[index] = 1
    return one_hot