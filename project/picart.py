import cv2
import numpy as np 

# reading the image
img = cv2.imread('nature.png')

# resizing the image 
# interpolation is cubic for best results
img_resized = cv2.resize(img, None, fx=1, fy=1)

# removing impurities from the image
img_cleared = cv2.medianBlur(img_resized, 3)
img_cleared = cv2.medianBlur(img_cleared, 3)
img_cleared = cv2.medianBlur(img_cleared, 3)

img_cleared = cv2.edgePreservingFilter(img_cleared, sigma_s = 5)

# bilateral image filtering
img_filtered = cv2.bilateralFilter(img_cleared, 3, 10, 5)

for i in range(2):
    img_filtered = cv2.bilateralFilter(img_filtered, 3, 20, 10)

for i in range(3):
    img_filtered = cv2.bilateralFilter(img_filtered, 5, 30, 10)

# sharpening the image using addWeighted()
gaussian_mask = cv2.GaussianBlur(img_filtered, (7,7), 2)
img_sharp = cv2.addWeighted(img_filtered, 1.5, gaussian_mask, -0.5, 0)
img_sharp = cv2.addWeighted(img_sharp, 1.4, gaussian_mask, -0.2, 10)

# dispaying image
cv2.imshow('final iamge', img_sharp)
cv2.imshow('original', img_resized)
cv2.imshow('cleared impurities', img_cleared)
cv2.waitKey(0)