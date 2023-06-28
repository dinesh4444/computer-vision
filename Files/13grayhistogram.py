# import libraries
import cv2
import numpy as np 
from matplotlib import pyplot as plt

img = cv2.imread('test.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([img], [0], None, [256], [0,255])
plt.plot(hist)

image_hist = cv2.equalizeHist(img)

hist = cv2.calcHist([image_hist], [0], None, [256], [0,255])
plt.plot(hist)


cv2.imshow('winname', img)
cv2.imshow('applied histogram', image_hist)
plt.show()