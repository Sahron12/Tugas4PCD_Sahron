#Butter Lowpass Filter
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Fungsi Butterworth Lowpass Filter
def butterworth_lowpass_filter(image, cutoff_frequency, order):
    # Konversi gambar ke domain frekuensi menggunakan Fast Fourier Transform (FFT)
    f = np.fft.fft2(image)
    # Shift frekuensi agar nol berada di tengah
    fshift = np.fft.fftshift(f)
    # Mendapatkan ukuran gambar
    rows, cols = image.shape
    # Mendapatkan koordinat tengah gambar
    center_row, center_col = rows // 2, cols // 2
    # Membuat filter lowpass Butterworth dengan jari-jari cutoff_frequency dan orde order
    y, x = np.ogrid[-center_row:rows-center_row, -center_col:cols-center_col]
    radius = np.sqrt(x*x + y*y)
    butterworth_filter = 1 / (1 + (radius / cutoff_frequency)**(2*order))
    # Mengalikan filter dengan gambar yang sudah di-shift
    filtered = fshift * butterworth_filter
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
order = 2
filtered_image = butterworth_lowpass_filter(image, cutoff_frequency, order)

# Menampilkan gambar asli dan hasil filter
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Gambar Asli')
plt.subplot(122), plt.imshow(filtered_image, cmap='gray'), plt.title('Hasil Filter Butterworth Lowpass')
plt.show()
