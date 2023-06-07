import cv2
import numpy as np
import imutils
import time
import playsound

# Assign variables for prototxt and model
prototxt = 'object_detection/mobilenet_ssd/MobileNetSSD_deploy.prototxt.txt'
model = 'object_detection/mobilenet_ssd/MobileNetSSD_deploy.caffemodel'
confThresh = 0.2

CLASSES = ["background","aeroplane","bicycle","bird","boat","bottle",
          "bus","car","cat","chair","cow","diningtable","dog","horse","motorbike",
          "person","pottedplant","sheep","sofa","train","tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
print('Loading model...')

# Loading model
net = cv2.dnn.readNetFromCaffe(prototxt, model)
print('Model Loaded')
print('Starting Camera feed')

cap = cv2.VideoCapture(0)
time.sleep(2.0)

def trigger_alarm():
    # Function to trigger an alarm or play a sound
    # You can customize this function to suit your requirements
    playsound.playsound('object_detection/mobilenet_ssd/alarm_sound.wav')

while True:
    _, frame = cap.read()
    frame = imutils.resize(frame, width=500)
    (h, w) = frame.shape[:2]
    image_resized = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(image_resized, 0.007843, (300,300), 127.5)
    net.setInput(blob)

    detections = net.forward()
    detshape = detections.shape[2]
    detected_classes = []
    for i in np.arange(0, detshape):
        confidence = detections[0,0,i,2]
        if confidence > confThresh:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] in ['person', 'cat', 'dog']:  # Specify the classes for triggering the alarm
                detected_classes.append(CLASSES[idx])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence*100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                if startY - 15 > 15:
                    y = startY - 15
                else:
                    y = startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    if 'person' in detected_classes or 'cat' in detected_classes or 'dog' in detected_classes:
        trigger_alarm()

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
