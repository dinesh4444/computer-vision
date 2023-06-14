# import libraries
import cv2
import numpy as np

# default trackbar function
def setValue(x):
    print("")

# creating the trackbars needed for adjusting the marker color
cv2.namedWindow('Color detectors')
cv2.createTrackbar("Upper Hue", "Color detectors", 153, 240, setValue)
cv2.createTrackbar("Upper Saturation", "Color detectors", 255, 255, setValue)
cv2.createTrackbar("Upper Value", "Color detectors", 255, 255, setValue)
cv2.createTrackbar("Lower Hue", "Color detectors", 64, 180, setValue)
cv2.createTrackbar("Lower Saturation", "Color detectors", 72, 255, setValue)
cv2.createTrackbar("Lower Value", "Color detectors", 49, 255, setValue)

# capture the input frame from webcam
def get_frame(cap, scaling_factor):
    ret, frame = cap.read()
    # resize the input frame
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    scaling_factor = 0.9
    # iterate until user presses ESC key
    while True:
        frame = get_frame(cap, scaling_factor)

        # convert HSV colorspace
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        u_hue = cv2.getTrackbarPos("Upper Hue", "Color detectors")
        u_saturation = cv2.getTrackbarPos("Upper Saturation", "Color detectors")
        u_value = cv2.getTrackbarPos("Upper Value", "Color detectors")
        l_hue = cv2.getTrackbarPos("Lower Hue", "Color detectors")
        l_saturation = cv2.getTrackbarPos("Lower Saturation", "Color detectors")
        l_value = cv2.getTrackbarPos("Lower Value", "Color detectors")

        # define color range in HSV colorspace
        Upper_hsv = np.array([u_hue, u_saturation, u_value])
        Lower_hsv = np.array([l_hue, l_saturation, l_value])

        # thresholding the HSV image to get only selected color
        mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
        # bitwise AND mask original image
        res = cv2.bitwise_and(frame, frame, mask=mask)
        res = cv2.medianBlur(res, 5)
        cv2.imshow("Original image", frame)
        cv2.imshow("Color Detector", res)
        # check if the user pressed ESC key
        key = cv2.waitKey(5)
        if key == 27:
            break
    cv2.destroyAllWindows()