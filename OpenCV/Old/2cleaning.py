import cv2
import numpy as np

def detect_longest_line(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a Canny edge detector to detect edges
    edges = cv2.Canny(gray, 50, 200, apertureSize=3)

    # Find lines in the image using the Hough transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    # Initialize variables to store the longest line and its length
    longest_line = None
    max_length = 0

    # Iterate over all the lines found in the image
    for line in lines:
        # Calculate the length of the current line
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        # If the length of the current line is greater than the maximum length seen so far,
        # update the maximum length and longest line variables
        if length > max_length:
            max_length = length
            longest_line = line

    # Draw a red mask on the image to highlight the longest line
    mask = np.zeros_like(gray)
    x1, y1, x2, y2 = longest_line[0]
    cv2.line(mask, (x1, y1), (x2, y2), (0, 0, 255), thickness=5)

    # Add the mask to the original image
    output = cv2.addWeighted(img, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)

    return output

img = cv2.imread('35Original.png')
output = detect_longest_line(img)
cv2.imshow('Longest Line', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
