import cv2
import numpy as np

def process_image(image_path, threshold):
    # Load the image
    image = cv2.imread(image_path)

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the image
    _, threshold_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY_INV)

    # Find the contours in the image
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the image
    for contour in contours:
        # Fit a line to the contour
        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Calculate the end points of the line
        lefty = int((-x * vy / vx) + y)
        righty = int(((gray_image.shape[1] - x) * vy / vx) + y)
        
        # Check if the points are within the valid range
        if 0 <= lefty < gray_image.shape[0] and 0 <= righty < gray_image.shape[0]:
            # Draw the line on the image (use black color)
            cv2.line(image, (image.shape[1] - 1, righty), (0, lefty), (0, 0, 0), 2)
        else:
            print(f'Skipping line with end points out of range: ({image.shape[1] - 1}, {righty}), (0, {lefty})')

    # Show the original and the processed image
    cv2.imshow('Processed Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

process_image('inverted_combined_mask.png', 10)
