import cv2
import numpy as np

# loading image 
image = cv2.imread('nature.png')

#resizing image
image = cv2.resize(image, (450,450))

# translation matrix
matrix = np.float32([[1,0,100],[0,1,100]])

# applying the matrix to the image
trnslated = cv2.warpAffine(image, matrix, (image.shape[1]+100, image.shape[0]+100))

# showing the image
cv2.imshow('translation',trnslated)
cv2.waitKey(0)