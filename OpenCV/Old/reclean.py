import cv2
import numpy as np

def is_triangle(contour):
    if len(contour) == 3:
        return True
    return False

def area_triangle(points):
    a = np.linalg.norm(points[0] - points[1])
    b = np.linalg.norm(points[1] - points[2])
    c = np.linalg.norm(points[2] - points[0])
    s = (a + b + c) / 2
    return np.sqrt(s * (s - a) * (s - b) * (s - c))

# Load the image
image = cv2.imread('inverted_combined_mask.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection using Canny function
edges = cv2.Canny(gray, 50, 150)

# Find contours in the binary image
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Initialize variables for the largest triangle
largest_triangle = None
largest_area = 0

# Iterate through the contours and detect triangles
for contour in contours:
    # Approximate the contour with a polygon
    polygon = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

    # Check if the polygon is a triangle
    if is_triangle(polygon):
        # Calculate the area of the triangle
        area = area_triangle(polygon)

        # Update the largest triangle if needed
        if area > largest_area:
            largest_area = area
            largest_triangle = polygon

# Create a mask for the largest triangle
height, width, _ = image.shape
mask = np.zeros((height, width), dtype=np.uint8)
cv2.fillPoly(mask, [largest_triangle], 255)

# Apply the original mask to the image
result = cv2.bitwise_and(image, image, mask=mask)

# Display the final image
cv2.imshow("Largest Triangle", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
