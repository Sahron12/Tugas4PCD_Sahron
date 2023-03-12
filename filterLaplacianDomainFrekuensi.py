import cv2
import numpy as np
from matplotlib import pyplot as plt

# Membaca gambar
img = cv2.imread('suneo.jpg', 0)

# Menghitung transformasi Fourier dari gambar
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Membuat kernel filter Laplacian
kernel = np.array([[0,-1,0], [-1,4,-1], [0,-1,0]])

# Mengaplikasikan kernel filter Laplacian ke domain frekuensi
kernel_dft = np.fft.fft2(kernel, s=img.shape)
kernel_dft_shift = np.fft.fftshift(kernel_dft)
filtered = fshift * kernel_dft_shift

# Menghitung inverse Fourier transform dari hasil filtering
result = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))

# Menampilkan gambar asli, kernel filter, dan hasil filtering
plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2), plt.imshow(kernel, cmap='gray')
plt.title('Laplacian Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3), plt.imshow(result, cmap='gray')
plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])
plt.show()
