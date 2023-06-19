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

def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# Load the image
image = cv2.imread('35Original.png')

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
        # Calculate the area of the contour (not the triangle)
        contour_area = cv2.contourArea(contour)

        # Update the largest triangle if needed
        if contour_area > largest_area:
            largest_area = contour_area
            largest_triangle = polygon

# Draw a red mask over the largest triangle
if largest_triangle is not None:
    cv2.drawContours(image, [largest_triangle], -1, (0, 0, 255), -1)

# Set the desired window size
window_width = 800
window_height = 600

# Resize the image to the desired window size
resized_image = resize_image(image, width=window_width, height=window_height)

# Display the final image
cv2.imshow("Largest Triangle", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
