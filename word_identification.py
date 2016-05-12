import cv2
from matplotlib import pyplot as plt

img = cv2.imread('KNMP.jpg') 

# convert to grayscale
gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imwrite( "Greyed_KNMP.jpg", gray_image)

# smooth the image to avoid noises
smoothed_image = cv2.medianBlur(gray_image,9)
cv2.imwrite( "Smoothed_KNMP.jpg", gray_image)


# Apply adaptive threshold
thresh = cv2.adaptiveThreshold(smoothed_image,255,1,1,11,2)
cv2.imwrite("Thresh_KNMP.jpg", thresh)

thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
cv2.imwrite("TC_KNMP.jpg", thresh_color)


# apply some dilation and erosion to join the gaps
thresh = cv2.dilate(thresh,None,iterations = 3)
thresh = cv2.erode(thresh,None,iterations = 2)


cv2.imwrite( "T_KNMP.jpg", thresh)


# Find the contours
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)


# For each contour, find the bounding rectangle and draw it
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)
    i = cv2.rectangle(thresh_color,(x,y),(x+w,y+h),(0,255,0),5)

cv2.imwrite("Identification.jpg", thresh_color)


