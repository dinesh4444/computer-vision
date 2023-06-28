# importing the module
import cv2
import numpy as np 

# import the image
img = cv2.imread('nature.png')

# Form thr filters
kernel_identity = np.array([[0,0,0],[0,1,0],[0,0,0]])
kernel_3 = np.ones((3,3), dtype=np.float32) / 9.0
kernel_11 = np.ones((11,11), dtype=np.float32) / 121.0

# apply the filter
output_1 = cv2.filter2D(img, -1, kernel_identity)
output_2 = cv2.filter2D(img, -1, kernel_3)
output_3 = cv2.filter2D(img, -1, kernel_11)

# show the image
cv2.imshow('same', output_1)
cv2.imshow('3 blur', output_2)
cv2.imshow('11 blur', output_3)
cv2.waitKey(0)