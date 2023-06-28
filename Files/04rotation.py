import cv2
import numpy as np

# loading the image
image = cv2.imread('nature.png')

height, width = image.shape[:2]

# translation matrix
matrix = cv2.getRotationMatrix2D((width/2, height/2), 10,1)

# Applying the matrix to the image
translation = cv2.warpAffine(image, matrix, (width, height))

# showing the image
cv2.imshow('translation', translation)
cv2.waitKey(0)