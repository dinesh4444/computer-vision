# import libraries
import cv2
import numpy as np 

# trackbar header
def changeRadius(value):
    global radius
    radius = value

# for changing the focus of the mask
def changeFocus(scope):
    global value 
    value = scope

# reading image and getting properties 
img = cv2.imread('man.jpg')
img = cv2.resize(img, (400, 600))
rows, cols = img.shape[:2]
value = 1
scope = 130
mask = np.ones((int(rows * (value * 0.1 + 1)), int(cols * (value * 0.1 + 1))))

cv2.namedWindow('Trackbars')
cv2.createTrackbar('Radius', 'Trackbars', 130, 500, changeRadius)
cv2.createTrackbar('Focus', 'Trackbars', 1, 10, changeFocus)

while True:
    # generating vignette mask using Gaussian kernels
    kernel_x = cv2.getGaussianKernel(int(cols * (0.1 * value+1)), radius)
    kernel_y = cv2.getGaussianKernel(int(rows* (0.1 * value+1)), radius)
    kernel = kernel_y * kernel_x.T

    # Normalizing the kernel
    kernel = kernel / np.linalg.norm(kernel)

    # generate a mask to image
    mask = 255 * kernel
    output = np.copy(img)
    # applying the mask to each channels in the input image
    mask_imposed = mask[int(0.1 * value * rows): , int(0.1 * value * cols):]
    for i in range(3):
        output[:,:,i] = output[:,:,i] * mask_imposed
    cv2.imshow('original', img)
    cv2.imshow('vignette', output)
    key = cv2.waitKey(50)
    if (key == ord('q')):
        break
    elif (key == ord('s')):
        cv2.imwrite('output/output_mask{}_deviation{}.jpg', output)