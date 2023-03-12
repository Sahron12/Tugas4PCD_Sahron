#Ideal Lowpass Filter
import numpy as np
import cv2
import matplotlib.pyplot as plt
# Fungsi Ideal Lowpass Filter
def ideal_lowpass_filter(image, cutoff_frequency):
    # Konversi gambar ke domain frekuensi menggunakan Fast Fourier Transform (FFT)
    f = np.fft.fft2(image)
    # Shift frekuensi agar nol berada di tengah
    fshift = np.fft.fftshift(f)
    # Mendapatkan ukuran gambar
    rows, cols = image.shape
    # Mendapatkan koordinat tengah gambar
    center_row, center_col = rows // 2, cols // 2
    # Membuat filter lowpass dengan jari-jari cutoff_frequency
    y, x = np.ogrid[-center_row:rows-center_row, -center_col:cols-center_col]
    radius = np.sqrt(x*x + y*y)
    mask = radius <= cutoff_frequency
    # Mengalikan filter dengan gambar yang sudah di-shift
    filtered = fshift * mask
    # Shift kembali gambar ke domain spasial
    filtered_shift = np.fft.ifftshift(filtered)
    # Konversi gambar kembali ke domain spasial menggunakan Inverse Fourier Transform (IFFT)
    filtered_image = np.fft.ifft2(filtered_shift)
    # Mengambil bagian real dari gambar hasil IFFT
    filtered_image = np.real(filtered_image)
    return filtered_image
# Membaca gambar
image = cv2.imread('suneo.jpg', 0) # Baca gambar dalam mode grayscale
# Melakukan filter
cutoff_frequency = 30
filtered_image = ideal_lowpass_filter(image, cutoff_frequency)
# Menampilkan gambar asli dan hasil filter
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Gambar Asli')
plt.subplot(122), plt.imshow(filtered_image, cmap='gray'), plt.title('Hasil Filter Ideal Lowpass')
plt.show()