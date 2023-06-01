# Computer Vision

## 1. OpenCV

- OpenCV is a open source programming library with real-time computer vision capabilities

### 1.1 Modules of opencv

- core - used for basic core functions, data structure, functionality to other modules.
- ImgProc - used for image processing-color spaces, geometrical transformation, histogram, image filtering.
- dnn - used for functionality for loading serialized network models.
- ML - userd for regression task, clusturing task in image , anomaly detection.
- video - used for video analysis including background substraction, motion estimation, and object tracking algorithms.
- highgui - create and manipulate windows that can display images.
- flann - fast library for approximate nearest neighbors-collection of algorithms that are highly suited for fast nearest neighbors search.
- photo - provides the functions of the computational photograpghy.
- stiching - implements a stiching pipeline that performs automatic panoromic image stiching.
- shape - shape distance and matching, can be used for shape matching, retrival or comparison.
- superres - this modeule contains a set of classes and methods that can be used for resolution enhancement.
- videostab - used for video stabilization.
- viz - 3D visualizer display widgets that provides several methods to interact with sence and widgets.
- imgcodecs - used for image codecs, image file read & write.
- objdetect - detection of objects and instances of predifined classes.
- features2d - used for 2D features framework. This modeule includes feature detector, descriptors and descriptor matches.
- calib3d - used for camera calibration 3D reconstruction, covers basic multiple view geometry algorithms, stereo corespondence algorithms, object pose estimation, both single and stereo camera calibration, and also 3D reconstruction.

### 1.2 Application of computer vision

- Feature matching
- 3D image stiching
- Egomotion Estimation
- Medical Image Analysis
- Human computer interface
- stereo vision
- segmentation & recognition
- structure from motion
- Augmented reality
- motion tracking

## Basic of Imge

#### what is an image?

- An image can be described as a 2D function, f(x,y) where (x,y) are the spetial coordinates and the value of f at any point (x,y) is proportional to the brightness or gray levels of the image.

#### Difference between image and digital image?

- The value of f(x,y) will be always a descrite value in digital image.

#### Grayscale image

- No color information. The pixel only contain the gray levels.
- single channel

#### RGB image

- color information is mandatory, the pixel to contain the RGB element.
- Three channel

## Color Spaces

- A color space is mathematical model describing the way color can be represented using tuples of numbers.

#### Major cknown color spaces

- RGB
- CMYK
- HSV
- LAB

## 3. Basic image function

#### image reading

    img = cv2.imread('image.png', cv2.IMREAD_COLOR)
    img1 = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('image.png', cv2.IMREAD_UNCHANGED)

#### display function

    cv2.imshow('show image', img1)

<img src="C:/Users/HP.DESKTOP-5IHNLMQ/Desktop/stud/gray/image.jpg" alt="grayscale image" width=500 height=400>

#### video basics

    import cv2

    # instance of video capture
    cap = cv2.VideoCapture(0)
    opened = cap.isOpened()

    # fourcc
    fourcc = cv2.VideoWriter_fourcc(\*'MJPG')

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

## Lines, shape, figures

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

## Fonts of opencv

    import cv2
    import numpy as np

    # creating canvas of 800x600 (three channels)
    canvas = np.zeros((800,600))

    # list of all fonts
    fonts = [cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_PLAIN,
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX]

    position = (10,30)
    for i in range(0,8):
        cv2.putText(canvas, "THIS IS OPENCV !", position, fonts[i], 1.1, (255,255,255), 2, cv2.LINE_AA)
        position = (position[0], position[1] + 30)
        cv2.putText(canvas, "THIS IS OPENCV !".lower(), position, fonts[i], 1.1, (255,255,255), 2, cv2.LINE_AA)
        position = (position[0], position[1] + 30)


    # Displaying the canvas
    cv2.imshow('fonts', canvas)
    cv2.waitKey(0)

## Rotation, Translaion, Scaling

- An Optical Zoom means moving the zoom lens so that it increases the magnification of light before it even reaches the digital sensor.
- A Digital Zoom is not really zoom, it is simply interpolating the image after it has been acquired at the sensor (pixilation process).

#### Scaling referes to changing the size i.e, increasing or decreasing the pixels in digital image

- It means Resampling an image and then assigning new gray values to the resampled positions.
- Types of interpollation
- 1. Linear interpolation
- 2. Area interpolation
- 3. cubic interpolation
- 4. Nearest neighbor interpolation
- 5. cynocidal interpolation

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

## Image Translation
- Shit an image in coordinate space by adding a specified value to the X and Y coordinates.
- Translation matrix(M)
- Apply M to image

    import cv2
    import numpy as np

    # loading image 
    image = cv2.imread('nature.png')

    #resizing image
    image = cv2.resize(image, (450,450))

    # translation matrix
    matrix = np.float32([[1,0,100],[0,1,100]])

    # applying the matrix to the image
    trnslated = cv2.warpAffine(image, matrix, (image.shape[1]+100, image.shape[0]+100))

    # showing the image
    cv2.imshow('translation',trnslated)
    cv2.waitKey(0)

## Transformation

#### What is Geometric transformation?
- Modify Spatial Relationship between pixels.
- Image can be shifted, rotated and strached in multiple ways.

#### 1. Euclidean or Isometric transformation
- whenever an image is shifted in x and y-axis, or rotate in pericular pixel.
- It has three degree of freedom.
#### Charecteristics:
- Distance remains preserved
- Angles remain preserved
- Shapes remain preserved

#### 2. Affine Transformation
- Has six degree of freedom , two for translation, one for rotation, one for scaling, one for scaling direction, and one for scaling ratio.
- The matrix can be rotated, translated, scaled, sheared.
- Parallel lines preserved but may be sheared. Square may become parallelogram.