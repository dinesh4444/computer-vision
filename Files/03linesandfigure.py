import cv2
import numpy as np 

# creating canvas 500x500 (Three channels)
canvas = np.zeros((500,500,3))

# Drawing line
#cv2.line(img, pt1, pt2, color, thickness, linetype)
cv2.line(canvas,(0,0), (100,100), (0,255,0), 2, cv2.LINE_4)
cv2.line(canvas,(0,50), (150,250), (0,0,255), 2, cv2.LINE_4)

# Types of lines 
#1. LINE_4 = bresenham algorithm
#2. LINE_8 = bresenham algorithm
#3. LINE_AA = gaussian filtering

# Drawing Rectangle
cv2.rectangle(canvas, (200,200), (250,270), (0,0,255), -1)

# Drawing a Circle
cv2.circle(canvas, (250,250), 10, (255,0,0), 3)

# Drawing a arrow line
cv2.arrowedLine(canvas, (400,400), (400,500), (255,255,255), 2)

# Polylines  
#required points we need to join
pts = np.array([[250,5],[220,80],[280,80],[100, 100], [250,250]], np.int32)

# reshape the points to shape (number_vertex, 1, 2)
pts = pts.reshape((-1,1,2))

#Draw the polyline
# Here Boolean True and False setermine if the figure is closed
cv2.polylines(canvas, [pts], True, (0,255,255), 3)

# Showing the canvas
cv2.imshow('output', canvas)
cv2.waitKey()