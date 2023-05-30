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

<img src="C:\Users\HP.DESKTOP-5IHNLMQ\Desktop\stud\gray image.jpg" alt="grayscale image" width=800 height=400>

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




