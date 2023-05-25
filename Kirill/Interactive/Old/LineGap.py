import cv2
import numpy as np

# Load the image
image = cv2.imread('inverted_combined_mask.png', 0)

# Threshold the image to binary for morphological operations
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Define a kernel for the dilation operation
# A larger size of the kernel will fill bigger gaps
kernel_size = 2
kernel = np.ones((kernel_size, kernel_size), np.uint8)

# Use dilation to fill the gaps
dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

# Save the result
cv2.imwrite('filled_image.png', dilated_image)

dilated_image = cv2.bitwise_not(dilated_image)

cv2.imwrite('filled_image2.png', dilated_image)
