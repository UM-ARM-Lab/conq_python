import cv2
import numpy as np

# Define a function to do nothing (used for the trackbars)
def nothing(x):
    pass

class PixelCoord:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class BoundingBox:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def top_left(self):
        return (self.x, self.y)

    def bottom_right(self):
        return (self.x + self.width, self.y + self.height)

# Load the image
# get using: python3 get_image.py 192.168.80.3 --image-sources left_fisheye_image --pixel-format PIXEL_FORMAT_RGB_U8
# image = cv2.imread('left_fisheye_image1.jpg')
image = cv2.imread('left_fisheye_image2.jpg')

# Define variables to store the best slider values
best_lower_h = 38
best_upper_h = 90
best_lower_s = 79
best_upper_s = 201
best_lower_v = 37
best_upper_v = 255

# Convert the image to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# Create a window to display the image
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.imshow('Original Image', image)

# Create a window to display the filtered image
cv2.namedWindow('Filtered Image', cv2.WINDOW_NORMAL)

# Create trackbars for the lower and upper range of H, S, and V channels
cv2.createTrackbar("L - H", "Filtered Image", best_lower_h, 179, nothing)
cv2.createTrackbar("U - H", "Filtered Image", best_upper_h, 179, nothing)
cv2.createTrackbar("L - S", "Filtered Image", best_lower_s, 255, nothing)
cv2.createTrackbar("U - S", "Filtered Image", best_upper_s, 255, nothing)
cv2.createTrackbar("L - V", "Filtered Image", best_lower_v, 255, nothing)
cv2.createTrackbar("U - V", "Filtered Image", best_upper_v, 255, nothing)

while True:
    # Get the current positions of the trackbars
    lower_h = cv2.getTrackbarPos("L - H", "Filtered Image")
    upper_h = cv2.getTrackbarPos("U - H", "Filtered Image")
    lower_s = cv2.getTrackbarPos("L - S", "Filtered Image")
    upper_s = cv2.getTrackbarPos("U - S", "Filtered Image")
    lower_v = cv2.getTrackbarPos("L - V", "Filtered Image")
    upper_v = cv2.getTrackbarPos("U - V", "Filtered Image")
    
    # Create a mask using the lower and upper range of H, S, and V channels
    lower_range = np.array([lower_h, lower_s, lower_v])
    upper_range = np.array([upper_h, upper_s, upper_v])
    mask = cv2.inRange(hsv_image, lower_range, upper_range)
    
    # Erode the images to get rid of noise
    erosion_size = 3
    el = cv2.getStructuringElement(cv2.MORPH_RECT, (erosion_size, erosion_size))
    eroded = cv2.erode(mask.astype(np.uint8), el)

    # Now dilate by a slightly larger kernel to close gaps
    dilation_size = erosion_size + 2
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))

    dilated = cv2.dilate(eroded, el)

    # Now get contours of the weeds.
    cnt, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(cnt))
    max_cnt = max(cnt, key=cv2.contourArea)

    # find the centroid pixel of the weed.
    M = cv2.moments(max_cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    contour_centroid = PixelCoord(cx, cy)

    # find the bounding box of the weed.
    x, y, w, h  = cv2.boundingRect(max_cnt)
    bbox = BoundingBox(x, y, w, h)
    
    # Apply the mask to the original image
    filtered_image = cv2.bitwise_and(image, image, mask=dilated)

    # Show the filtered image
    cv2.rectangle(image, (bbox.x, bbox.y), (bbox.x + bbox.width, bbox.y + bbox.height), (255,0,0), 4)
    cv2.imshow('Original Image', image)

    # cv2.imshow('Filtered Image', filtered_image)
    cv2.imshow('Filtered Image', dilated)
    
    # Exit the program if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()