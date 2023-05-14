import cv2
import numpy as np

def main():
    # Load the image
    image_path = 'inverted_combined_mask.png'
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours, giving each an ID, and draw them on the original image using a mask
    for i, contour in enumerate(contours):
        cv2.drawContours(img, [contour], 0, (0, 0, 255), 2)
        x, y, _, _ = cv2.boundingRect(contour)
        cv2.putText(img, f'ID: {i}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Create a resizable window
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)

    # Display the result
    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

