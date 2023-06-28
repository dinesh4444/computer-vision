# import the libraries
import cv2
import numpy as np 

# read image
img = cv2.imread('women.jpg')
img = cv2.resize(img, (600, 800))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# performing the edge detection
gradient_sobelx = cv2.Sobel(img, -1, 1, 0)
gradient_sobely = cv2.Sobel(img, -1, 0, 1)
gradient_sobelxy = cv2.addWeighted(gradient_sobelx, 0.5, gradient_sobely, 0.5, 0)

gradient_laplacian = cv2.Laplacian(img, -1)

canny_output = cv2.Canny(img, 80, 150)

cv2.imshow('sobel x', gradient_sobelx)
cv2.imshow('sobel y', gradient_sobely)
cv2.imshow('sobel xy', gradient_sobelxy)
cv2.imshow(' laplacian', gradient_laplacian)
cv2.imshow('canny', canny_output)
cv2.waitKey(0)