import numpy as np 
import cv2

image = np.zeros((500,500))
image[:,:] = 100

image = image[:,:] + 10

image[200:300, 200:300] = 255

cv2.imwrite('output/sample.jpg', image)
cv2.imshow('output/sample.jpg', image)
cv2.waitKey(0)

