# import libraries
import cv2
import numpy as np

kernel = np.ones((3,3))

# compute the frame difference
def frame_diff(prev_frame, curr_frame, next_frame):
    diff_frame1 = cv2.absdiff(next_frame, curr_frame)
    # absolute difference between current frame and previous frame
    diff_frame2 = cv2.absdiff(curr_frame, prev_frame)
    # return the result of bitwise 'AND' between the above two resultant images
    return cv2.bitwise_and(diff_frame1, diff_frame2)

# capture the frame from webcam
def get_frame(cap):
    ret, frame = cap.read()
    # resize the image
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    # return the grayscale image
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    scaling_factor = 0.9
    prev_frame = get_frame(cap)
    curr_frame = get_frame(cap)
    next_frame = get_frame(cap)
    # iterate until the user presses the ESC key
    while True:
        frame_difference = frame_diff(prev_frame, curr_frame, next_frame)
        _, frame_th = cv2.threshold(frame_difference, 0, 255, cv2.THRESH_TRIANGLE)
        #frame_th = cv2.dilate(frame_th, kernel)
        cv2.imshow("Object Movement", frame_difference)
        cv2.imshow("object", frame_th)

        # update the variables
        prev_frame = curr_frame
        curr_frame = next_frame
        next_frame = get_frame(cap)
        # check if the user pressed ESC
        key = cv2.waitKey(5)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()