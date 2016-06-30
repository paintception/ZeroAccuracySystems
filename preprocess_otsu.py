import cv2
from PIL import Image
import time
import os

def otsufy_pillow_image(image):
    file_path = "tmp" + str(int(time.time()*1000000)) + ".png"
    image.save(file_path)
    otsufy_file(file_path)
    image = Image.open(file_path)
    image.load()
    os.remove(file_path)
    return image

def otsufy_file(file_path):
    img = cv2.imread(file_path,0)
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3, th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(file_path,th3)