# import libraries
import cv2
import numpy as np 

# reading image and getting properties
img = cv2.imread('man.jpg')
img = cv2.resize(img, (600,800))
rows, cols = img.shape[:2]

# generating vigenette mask using Gaussian kernels
kernel_x = cv2.getGaussianKernel(cols, 200)
kernel_y = cv2.getGaussianKernel(rows, 200)
kernel = kernel_y * kernel_x.T 

# normalizing the kernel
kernel = kernel / np.linalg.norm(kernel)

# genrating a mask to image
mask = 255 * kernel
output = np.copy(img)

# applying the mask to each channel in the input image
for i in range(3):
    output[:,:,i] = output[:,:,i] * mask
cv2.imshow('original', img)
cv2.imshow('Vignette', output)
cv2.waitKey(0)