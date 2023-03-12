import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image in grayscale
img = cv2.imread('suneo.jpg', 0)

# Perform FFT
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Compute magnitude spectrum
mag_spectrum = 20 * np.log(np.abs(fshift))

# Display the original and magnitude spectrum
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(mag_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
