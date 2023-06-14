import cv2
import numpy as np 
from matplotlib import pyplot as plt 

img = cv2.imread('julia.png')
b, g, r = cv2.split(img)

cv2.imshow('winname', img)
hist = cv2.calcHist([b], [0], None, [256], [0,255])
plt.plot(hist)
hist = cv2.calcHist([g], [0], None, [256], [0,255])
plt.plot(hist)
hist = cv2.calcHist([r], [0], None, [256], [0,255])
plt.plot(hist)
plt.show()