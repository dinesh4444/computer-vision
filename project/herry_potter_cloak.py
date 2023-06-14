# import the library
import cv2
import numpy as np 

# fuction for the calling trackbar
def hello(x):
    print("")

# intialization of camera
cap = cv2.VideoCapture(0)
bars = cv2.namedWindow("bars")

# create trackbar to calculate HSV upper and lower values
cv2.createTrackbar("Upper Hue", "bars", 110, 180, hello)
cv2.createTrackbar("Upper Saturation", "bars", 255, 255, hello)
cv2.createTrackbar("Upper Value", "bars", 255, 255, hello)
cv2.createTrackbar("Lower Hue", "bars", 68, 180, hello)
cv2.createTrackbar("Lower Saturation", "bars", 55, 255, hello)
cv2.createTrackbar("Lower Value", "bars", 54, 255, hello)

# caputing the initial frames for creation of background
while True:
    cv2.waitKey(1000)
    ret, init_frame = cap.read()
    # check if the frame is returened then break
    if ret:
        break

# start capturing the frame for actual magic
while True:
    ret, frame = cap.read()
    inspect = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # getting the HSV values for masking the cloak
    upper_hue = cv2.getTrackbarPos("Upper Hue", "bars")
    upper_saturation = cv2.getTrackbarPos("Upper Saturation", "bars")
    upper_value = cv2.getTrackbarPos("Upper Value", "bars")
    l_hue = cv2.getTrackbarPos("Lower Hue", "bars")
    l_saturation = cv2.getTrackbarPos("Lower Saturation", "bars")
    l_value = cv2.getTrackbarPos("Lower Value", "bars")

    # kernel to be used for dilation
    kernel = np.ones((3,3), np.uint8)

    upper_hsv = np.array([upper_hue, upper_saturation, upper_value])
    lower_hsv = np.array([l_hue, l_saturation, l_value])

    mask = cv2.inRange(inspect, lower_hsv, upper_hsv)
    mask = cv2.medianBlur(mask, 3)
    mask_inv = 255 - mask
    mask = cv2.dilate(mask, kernel, 5)

    # the mixing of framein a combination to achieve the required frame
    b = frame[:,:,0]
    g = frame[:,:,1]
    r = frame[:,:,2]
    b = cv2.bitwise_and(mask_inv, b)
    g = cv2.bitwise_and(mask_inv, g)
    r = cv2.bitwise_and(mask_inv, r)
    frame_inv = cv2.merge((b,g,r))

    b = init_frame[:,:,0]
    g = init_frame[:,:,1]
    r = init_frame[:,:,2]
    b = cv2.bitwise_and(b, mask)
    g = cv2.bitwise_and(g, mask)
    r = cv2.bitwise_and(r, mask)
    blanket_Area = cv2.merge((b,g,r))

    final = cv2.bitwise_or(frame_inv, blanket_Area)

    cv2.imshow("Harry cloak", final)
    cv2.imshow("original", frame)

    if (cv2.waitKey(3) == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()

