import cv2

# instance of video capture
cap = cv2.VideoCapture(0)
opened = cap.isOpened()

# fourcc
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

fps = cap.get(cv2.CAP_PROP_FPS)

# video writer
out = cv2.VideoWriter('output/jj.mp4', fourcc, fps, (int(width), int(height)))

print("Frames are {}".format(fps))
print(height)

if (opened):
    while (cap.isOpened()):
        ret, frame = cap.read()
        if (ret==True):
            cv2.imshow('window', frame)
            if (cv2.waitKey(2) == 27):
                break

out.release()
cap.release()
cv2.destroyAllWindows()