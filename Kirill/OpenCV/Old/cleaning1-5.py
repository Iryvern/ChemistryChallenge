import cv2
import numpy as np

def find_longest_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    longest_contour = None
    max_length = 0

    for contour in contours:
        length = cv2.arcLength(contour, True)
        if length > max_length:
            max_length = length
            longest_contour = contour

    return longest_contour

def create_mask(img, contour):
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [contour], 0, (0, 0, 255), 2)
    return mask

def show_image(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if img is None:
        print("Error: Image not found or unable to open.")
        return

    # Find the longest continuous black line
    longest_contour = find_longest_contour(img)

    if longest_contour is not None:
        # Create a mask with the longest line
        mask = create_mask(img, longest_contour)

        # Show the mask image
        cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
        cv2.imshow("Mask", mask)
    else:
        # Show an error message if no line is found
        print("Error: No line found in image.")

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage:
image_path = "35Original.png"
show_image(image_path)
