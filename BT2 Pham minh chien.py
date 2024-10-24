import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh dưới dạng ảnh xám (grayscale)
image_path = "C:/Users/Public/gdg.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Toán tử Sobel
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel theo hướng x
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel theo hướng y

sobel_combined = cv2.magnitude(sobel_x, sobel_y)  # Độ lớn của gradient

# Toán tử Laplacian of Gaussian (LoG)
gaussian_blur = cv2.GaussianBlur(image, (3, 3), 0)  # Làm mịn Gaussian
log = cv2.Laplacian(gaussian_blur, cv2.CV_64F)  # Laplacian sau khi làm mịn Gaussian

# Hiển thị kết quả
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Ảnh gốc')
plt.imshow(image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Dò biên Sobel')
plt.imshow(sobel_combined, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Laplacian of Gaussian (LoG)')
plt.imshow(log, cmap='gray')

plt.show()
