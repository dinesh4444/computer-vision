import cv2
import numpy as np 
import random
import winsound

net = cv2.dnn.readNetFromDarknet('object_detection/yolov3/yolov3.cfg', 'object_detection/yolov3/yolov3.weights')                           

# define the classes that yolo can detect
classes = []
with open('object_detection/yolov3/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# set up the alarm
def trigger_alarm():
    # play system alert sound
    frequency = 2500
    duration = 2000
    winsound.Beep(frequency, duration)
    print('Alarm triggered')

# generate random colors for each class
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# load the video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, frame = cap.read()

    # detect objects in the frame
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers)

    # process the detections
    boxes = []
    confidences = []
    class_ids = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])

                # calculate bounding box coordinates
                x = int(center_x - (width/2))
                y = int(center_y - (height/2))

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # apply non-maximum supression to remove overlapping detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # draw the bounding boxes and labels
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box
            label = classes[class_ids[i]]
            color = colors[class_ids[i]].tolist()

            if label == 'person' or label == 'dog' or label == 'cat':
                trigger_alarm()

            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # display the frame
    cv2.imshow('object detection', frame)

    # exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
cap.release()
cv2.destroyAllWindows()