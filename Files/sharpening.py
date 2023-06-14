import cv2
import numpy as np 

# reading an image
img = cv2.imread('nature.png')

# gaussian kernel for sharpening
gaussian_blur = cv2.GaussianBlur(img, (7,7), 2)

# sharpening using addWeighted()
sharpened1 = cv2.addWeighted(img, 1.5,  gaussian_blur, -0.5, 0)
sharpened2 = cv2.addWeighted(img, 3.5,  gaussian_blur, -2.5, 0)
sharpened3 = cv2.addWeighted(img, 7.5,  gaussian_blur, -6.5, 0)

# showing the sharpened image
cv2.imshow('sharpende 1', sharpened1)
cv2.imshow('sharpened 2', sharpened2)
cv2.imshow('sharpened 3', sharpened3)
cv2.imshow('original', img)
cv2.waitKey(0)