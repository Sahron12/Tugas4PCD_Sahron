from PIL import Image, ImageFilter

# Load the image
image = Image.open('suneo.jpg')

# Apply Gaussian blur to the image
blurred = image.filter(ImageFilter.GaussianBlur(radius=3))

# Calculate the sharpened image using the unsharp masking technique
unsharp_image = Image.blend(image, blurred, 1.5)

# Display the original and sharpened images
image.show()
unsharp_image.show()

sharp = gray.astype(np.float32) + (gray.astype(np.float32) - smoothed.astype(np.float32))

# Menggabungkan gambar asli dan gambar yang di-sharpen
unsharp_masked = cv2.convertScaleAbs(img.astype(np.float32) + 1.5 * sharp.astype(np.float32))

# Menampilkan gambar hasil Unsharp Masking
cv2.imshow('Unsharp Masked Image', unsharp_masked)
cv2.waitKey(0)
cv2.destroyAllWindows()
