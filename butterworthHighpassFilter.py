import numpy as np
import matplotlib.pyplot as plt

# Membuat filter butterworth highpass
def butterworth_hp_filter(rows, cols, D0, n):
    P, Q = rows//2, cols//2
    H = np.zeros((rows, cols))
    for u in range(rows):
        for v in range(cols):
            D_uv = np.sqrt((u-P)**2 + (v-Q)**2)
            if D_uv != 0:
                H[u, v] = 1 / (1 + (D0/D_uv)**(2*n))
    return H

# Membaca gambar
img = plt.imread('suneo.jpg')

# Konversi menjadi grayscale
gray = np.mean(img, axis=2)

# Mendapatkan ukuran gambar
rows, cols = gray.shape

# Mendapatkan filter butterworth highpass
D0 = 50
n = 2
H = butterworth_hp_filter(rows, cols, D0, n)

# Konvolusi gambar dengan filter
G = np.fft.ifft2(np.fft.fft2(gray) * H).real

# Menampilkan gambar asli dan hasil filtering
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(G, cmap='gray')
plt.title('Butterworth Highpass Filter')
plt.axis('off')

plt.show()
