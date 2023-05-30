import cv2

# Channel mode is BGR in color mode
image = cv2.imread('nature.png', cv2.IMREAD_GRAYSCALE)

# function to display
cv2.imshow('output image', image)

cv2.waitKey()