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

## 4. Lines, shape, figures

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

## 5. Fonts of opencv

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

## 6. Rotation, Translaion, Scaling

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

#### Linear and Cubic Interpolation

    import cv2
    import numpy as np

    # scaling operation
    #cReading original image
    image = cv2.imread('nature.png')
    image_sized = cv2.resize(image, (200,200))

    # resizing the image using linear interpolation
    image_resized = cv2.resize(image, None, fx=1.1, fy=1.1, interpolation=cv2.INTER_LINEAR)

    # Resizing the image using cubic interpolation
    image_re_cube = cv2.resize(image, None, fx=1.1, fy=1.1, interpolation=cv2.INTER_CUBIC)

    #showing all three images
    cv2.imshow('Linear', image_resized)
    cv2.imshow('Cubic', image_re_cube)
    cv2.imshow('sized', image_sized)
    cv2.imshow('Original', image)
    if (cv2.waitKey() == ord('q')):
        cv2.destroyAllWindows()

#### Image Translation

- Shit an image in coordinate space by adding a specified value to the X and Y coordinates.
- Translation matrix(M)
- Apply M to image



    import cv2
    import numpy as np

    #loading image
    image = cv2.imread('nature.png')

    #resizing image
    image = cv2.resize(image, (450,450))

    #translation matrix
    matrix = np.float32([[1,0,100],[0,1,100]])

    #applying the matrix to the image
    trnslated = cv2.warpAffine(image, matrix, (image.shape[1]+100, image.shape[0]+100))
    #showing the image
    cv2.imshow('translation',trnslated)
    cv2.waitKey(0)

## 7.Transformation

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

#### 3. Projective Transformation
- In the projective transform, you change the projection of image.

## 8. Convolution and Filtering
- Convolution is a fundamental operation in image processing. We basically apply a mathematical operator to each pixel and change its value in some way. To apply this mathematical operator, we use another matrix called kernel.
- The kernel is fixed with its center on each pixel, and corresponding pixel are multiplied. The pixel values is replaced with sum of all multiplication.
- The kernel is called the "image filter" and the process of applying this kernel to the given image is called "image filtering". The output obtained after applying the kernel to the image is called the "filtering image".

### High pass and Lowpass filter
- What is frequency in Image?
- Frequency refers to the rate of change of pixel values. So we can say that the sharp edges would be high frequency contact because the pixel values change rapidly in that region. Going by that logic, Plain areas would be low frequency content.
#### Low pass filter
- Low pass filter is the type of frequency domain filter that attenuate the high frequency components and preserves the low frequency components.
#### High pass filter
- High pass filter is the type of frequency domain filter that attenuate the low frequency components and preserves the high frequency components.

#### Application of filtering
- Note: Remeber normalising filters before applying to image for stable results.
    output = cv2.filter2D(src, ddepth, kernel, anchor, border_type)

    import cv2                                               # importing the module
    import numpy as np 
    img = cv2.imread('nature.png')                          # import the image
    kernel_identity = np.array([[0,0,0],[0,1,0],[0,0,0]])      # Form thr filters
    kernel_3 = np.ones((3,3), dtype=np.float32) / 9.0
    kernel_11 = np.ones((11,11), dtype=np.float32) / 121.0     # apply the filter
    output_1 = cv2.filter2D(img, -1, kernel_identity)
    output_2 = cv2.filter2D(img, -1, kernel_3)
    output_3 = cv2.filter2D(img, -1, kernel_11) 
    cv2.imshow('same', output_1)         # show the image
    cv2.imshow('3 blur', output_2)
    cv2.imshow('11 blur', output_3)
    cv2.waitKey(0)

--------------------------------------------------------------------------------------------------

## Edge Detection

- The process of detection involves detecting sharp edges in the image and producing a binary image as the output. Typically, we draw white lines on a black background to indicate those edges.

#### Types of edge selection
- Sobel Edge filters (Sobal x and Sobal Y)
- Scharr edge Filters
- Laplacian Filters

#### Sobel Edge Filter
- The sobel operator computes an approximation of the gradient of an image intensity function. it depends on first order derivatives.
- Demerits : signal to noise ratio, not accurate results and discontinuity.
    cv2.Sobel(src, dst, ddepth, dx, dy)

#### Scharr Edge Filter
- The scharr operator is belived to give better results than sobel. the scharr operator is dependent on first order derivatives.
    cv2.Scharr(src, dst, ddepth, dx, dy)

#### Laplacian Filter
- Laplacian operator is also a derivative operator which is used to find edges in an image.
- The major difference between Laplacian and other operators like Prewitt, Sobel, Robinson and Kirsch is that these all are first order derivative masks but Laplacian is a second order derivative mask.
    cv2.Laplacian(src_gray, ddepth, ksize=kernel_size)

-------------------------------------------------------------------------------------------------

## Canny Edge detection
#### Steges of canny edge detection
- 1. Noise Reduction
- 2. Gradient Calculation
- 3. Non-Maximum Supression
- 4. Double thresholding
- 5. Edge tracking by Hysteresis

- 1. Noise Reduction
Since edge detection is susceptible to noise in the image, first step to remove the noise in the image with a Gaussian filter.
- 2. Gradient Calculation
Smoothened image is then filtered with a Sobel kernel in both X and Y direction to get derivatives of Gx and Gy. 
- 3. Non-Maximum Supression
The final image should have thin edges. Thus, we must perform non-maximum suppression to thin out the edges. The algorithm goes through all the points on the image and finds the pixels with the maximum value of gradient in the edge direction.
- 4. Double Thresholding
For this step we need two threshold values, minVal and maxVal. any edge with intensity gradient more than maxVal are sure to be edges and those below minVal are non-edges. Those who lies between these two thresholds are classified edges or non-edges based on their connectivity.
- 5. Edge tracking by Hysteresis
If they are connected to "sure-edge" pixels, they are considered to be part of edges. Otherwise, they are also discarded.

-------------------------------------------------------------------------------------------

## What is morphology and Morphological Transformation?
- Morphology is a broad set of image processing operations that process binary images based on structuring element or kernel which decides the nature of opertion.
- In a morphological operation, each pixel in the image is adjusted based on the value of each pixels in its neighborhood.
#### Types of morphological operations
- 1. Dilation
- 2. Erosion
- 3. Opening
- 4. Closing
- 5. Gradient
- 6. Top Hat
- 7. Black Hat

- Erosion Operaion
- A pixel in the original image (either 255 or 0) will be considered 255 only if all the pixels under the kernel is 255, otherwise it is eroded (made to zero).

- Dilation Operation
- Just opposite to erosion, Here a pixel element is '255' , if atleast one pixel under the kernel is '255'

- Opening Operation
- Many times used in Noise Removal, it is operation erosion followed by dilation.

- closing Operation
- Filling patches in the foreground object mask, it is operation dilation followed by erosion.

- Gradient operation.
- To find outlines of ojects.

- Top Hat
- Difference between input image and its opening, Highlights minor details in image (only)

- Black Hat
- To find bright objects on dark background

    kernel = np.ones((5,5), np.unit8)
    or
    cv2.getStructureingElement(cv2.MORPH_RECT, (5,5))
    erosion = cv2.erode(img, kernel, iterations = 1)
    dilation = cv2.dilate(img, kernel, iterations = 1)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

------------------------------------------------------------------------------------------------
## What is Image histogram ?
- An image histogram is a type of histogram that reflects the intensity (tonal) distribution of the image plotting the number of pixels for each intensity value.

#### What is Brightness?
- The brightness of a grayscale image can be defined as the average intensity of all the pixels of the image. 

#### Histogram Equalization
- The cv2.equalizeHist() function is used to eqalize the image histogram which normalizes the brightness and also incresases the contrast of the image. 

## CLAHE - Contrast Limited Adaptive Histogram
- CLAHE solves problem of impurity maximisation by Cliping the extra values. If any histogram bin is above the specified contrast limit, those pixels are clipped and distributed unformly to other bins before applying histogram equalization.

## Image segmentation
- Image segmentation is to modify the represenation of an image into another represenation that is easier to process. For example, image segmentation is commonly used to extract objects from the background based on some properties of the object.

#### Thresholding
- Thresholding is easiest form of image segmentation based on intensity values of pixel.

#### Types of thresholding
- 1. Global Thresholding : Manual, Otsu, Triangle
- 2. Adaptive thresholding : Mean, Gaussian, Niblack

## Color based Object Tracking
- Object tracking is a computer vision technique for locating position of objects in images or videos. or simply tracking an object in a live video.
- Color segmentation based.
- Frame differencing based.
- Feature matching based.
- Machine learning based.

------------------------------------------------------------------------------------------------------

## Contour
- Contour is a boundry around something that has well defined edges, so the machine is able to calculate difference in gradient, and form a recognisable shape through continuing change and draw a boundry around it.
- Through contour detection, we can detect the outlines (border) of objects, and localize them. Many exmples of contour detection include foreground extraction, image segmentation, detection and recognition.

    cv2.findCountours(img, mode, method)
    cv2.drawCountours(img, countours, countourldx, color, thickness)

#### contour Retrival modes
- 1. Retrive External : outputs only external the countours.
- 2. Retrive List : outputs all the contours without any hierarchy relationship.
- 3. Retrive Tree : outputs all the contours by establishing a hierarchy relationship.

#### Comparission methods
- Detected contours can be compressed to reduce the number of points. In this sense, Opencv provides several methods to reduce the number of points. This can be set with the parameter methods.
- CHAIN_APPROX_NONE : all boundry points are stored; no compression
- CHAIN_APPROX_SIMPLE 
- CHAIN_APPROX_TC89_L1
- CHAIN_APPROX_TC89_KCOS

