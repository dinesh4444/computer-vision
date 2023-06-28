# import libraries
import cv2
import numpy as np

# read the image and convert it to grayscale
img = cv2.imread('ball.jpg')
img = cv2.resize(img, None, fx=0.8, fy=0.8)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# now convert grayscale image to binary image
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# now convert the grayscale image to binary image
contours, hierarchy = cv2.findContours(binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

# visualize the data structure
print("Lenght od countours {}".format(len(contours)))
print(contours)

# draw countours on the original image
image_copy = img.copy()
image_copy = cv2.drawContours(image_copy, contours, -1, (0,255,0), thickness=2, lineType=cv2.LINE_AA)

# visualize the results
cv2.imshow("Grayscale image", gray)
cv2.imshow("Draw countours", image_copy)
cv2.imshow('Binary image', binary)

cv2.waitKey(0)
cv2.destroyAllWindows()