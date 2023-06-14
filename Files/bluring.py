import cv2
import numpy as np

# reading image from computer and taking dimensions
img = cv2.imread('nature.png')
rows, cols = img.shape[:2]

# kernel blurring
kernel = np.ones((25,25), np.float32) / 625.0
output_kernel = cv2.filter2D(img, -1, kernel)

# box filter and blur function blurring
output_blur = cv2.blur(img, (25,25))
output_box = cv2.boxFilter(img, -1, (5,5), normalize=False)

# gaussian blur
output_gaus = cv2.GaussianBlur(img, (5,5), 0)

# median blur (reduction of noise)
output_median = cv2.medianBlur(img, 5)

# bilateral filtering (reduction of noise and preserving of edges)
output_bilateral = cv2.bilateralFilter(img, 5, 6, 6)


cv2.imshow('kernel blurring', output_kernel)
cv2.imshow('box filter', output_box)
cv2.imshow('gaussian blur', output_gaus)
cv2.imshow('median blur', output_median)
cv2.imshow('bilateral filter', output_bilateral)
cv2.waitKey(0)

