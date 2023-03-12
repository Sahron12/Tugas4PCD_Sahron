import cv2
import numpy as np
from matplotlib import pyplot as plt

# Membaca gambar
img = cv2.imread('suneo.jpg', 0)

# Menghitung transformasi Fourier dari gambar
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Membuat filter highpass untuk mempertahankan fitur frekuensi tinggi
filter_high = np.zeros(fshift.shape, np.uint8)
filter_high[fshift > np.max(fshift)/2] = 1

# Membuat filter lowpass untuk mempertahankan fitur frekuensi rendah
filter_low = np.zeros(fshift.shape, np.uint8)
filter_low[fshift < np.max(fshift)/10] = 1

# Mengaplikasikan filter ke domain frekuensi
filtered_high = fshift * filter_high
filtered_low = fshift * filter_low

# Menghitung inverse Fourier transform dari hasil filtering
result_high = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_high)))
result_low = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_low)))

# Menampilkan gambar asli dan hasil filtering
plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2), plt.imshow(result_high, cmap='gray')
plt.title('Highpass Filtered Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3), plt.imshow(result_low, cmap='gray')
plt.title('Lowpass Filtered Image'), plt.xticks([]), plt.yticks([])
plt.show()
