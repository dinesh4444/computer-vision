import cv2
import numpy as np

# scaling operation
# Reading original image

image = cv2.imread('nature.png')

image_sized = cv2.resize(image, (200,200))
# resizing the image using linear interpolation
image_resized = cv2.resize(image, None, fx=1.1, fy=1.1, interpolation=cv2.INTER_LINEAR)

# Resizing the image using cubic interpolation
image_re_cube = cv2.resize(image, None, fx=1.1, fy=1.1, interpolation=cv2.INTER_CUBIC)

# showing all three images
cv2.imshow('Linear', image_resized)
cv2.imshow('Cubic', image_re_cube)
cv2.imshow('sized', image_sized)
cv2.imshow('Original', image)

if (cv2.waitKey() == ord('q')):
    cv2.destroyAllWindows()