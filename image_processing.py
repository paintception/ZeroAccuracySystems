import cv2
import numpy as np
from matplotlib import pyplot as plt
 
img = cv2.imread('A.png',0)
singleA = cv2.imread('A.png',0)

# global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
     
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th1,
           img, 0, th2,
           blur, 0, th3]

titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
           'Original Noisy Image','Histogram',"Otsu's Thresholding",
           'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
 
for i in xrange(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.savefig('Figure_One')
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.savefig('Figure_Two')
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    plt.savefig('Figure_Three')
    #plt.show()


def rotate_image_clockwise(image,angle):
    
    rows, cols = image.shape
    
    for i in range (angle):
        M = cv2.getRotationMatrix2D((cols/2,rows/2),i,1)
        dst = cv2.warpAffine(img,M,(cols,rows))
        #plt.imshow(dst)
        #plt.show()

def rotate_image_anticlockwise(image,angle):

    rows, cols = image.shape
    
    for i in range (angle):
        M = cv2.getRotationMatrix2D((cols/2,rows/2),i,1)
        dst = cv2.warpAffine(img,M,(cols,rows))   
        #plt.imshow(dst)
        #plt.show()
        
def flip_axis(image):   #FIXME NOT SURE IF THIS IS SUITABLE FOR THE NEURAL NETWORK
    
    flippedImage=cv2.flip(image,1)
    plt.imshow(flippedImage)
    plt.show()

rotate_image_clockwise(singleA,-20)
rotate_image_anticlockwise(singleA, 20)
flip_axis(singleA)

