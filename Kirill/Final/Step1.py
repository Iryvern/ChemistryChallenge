import cv2
import numpy as np

def find_biggest_shape(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(grayscale, 30, 100)

    # Find contours in the image
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours were found, return original image
    if not contours:
        print("No shapes were found in the image.")
        return image

    # Otherwise, find the largest contour
    biggest_contour = max(contours, key=cv2.contourArea)

    # Create an empty black mask of the same size as the image
    mask = np.zeros_like(grayscale)

    # Draw the largest contour on the mask in white
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)

    # Apply the mask to the image (this will keep only the pixels of the image that are white on the mask)
    image = cv2.bitwise_and(image, image, mask=mask)

    # Save the resulting image
    cv2.imwrite(output_path, image)

    # Create a named window, make it resizable
    cv2.namedWindow('Biggest Shape', cv2.WINDOW_NORMAL)
    
    # Get the screen size
    screen_res = 1280, 720  # You can replace this with your screen resolution
    scale_width = screen_res[0] / image.shape[1]
    scale_height = screen_res[1] / image.shape[0]
    scale = min(scale_width, scale_height)

    # Resized window width and height
    window_width = int(image.shape[1] * scale)
    window_height = int(image.shape[0] * scale)

    # cv2.resizeWindow() to resize image to fit on screen
    cv2.resizeWindow('Biggest Shape', window_width, window_height)

    # Show the image with the biggest shape highlighted
    cv2.imshow('Biggest Shape', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    find_biggest_shape('Graph.png', 'OutputSmaller.png')
