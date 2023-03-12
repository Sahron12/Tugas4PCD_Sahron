import cv2

# Load the image
img = cv2.imread('lenna.jpg')

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply median filter with kernel size 3
filtered_img = cv2.medianBlur(gray_img, 7)

# Show the original and filtered image
cv2.imshow('Original Image', gray_img)
cv2.imshow('Filtered Image', filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()