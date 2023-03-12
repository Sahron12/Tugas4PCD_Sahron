#Max Filter
import cv2
import numpy as np


# Load the image
img = cv2.imread('suneo.jpg')

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply max filter with kernel size 3
kernel = np.ones((3,3),np.uint8)
filtered_img = cv2.dilate(gray_img, kernel)

# Show the original and filtered image
cv2.imshow('Original Image', gray_img)
cv2.imshow('Filtered Image', filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
