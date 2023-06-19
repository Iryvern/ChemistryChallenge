import cv2
import numpy as np
import random

def random_warm_color():
    """Generate a random warm RGB color."""
    
    blue = random.randint(0, 40)  
    green = random.randint(0, 255)  
    red = random.randint(100, 255)
    
    return blue, green, red

def flood_fill(img, x, y, new_color):
    # Stack of pixels that need to be checked
    stack = [(x, y)]

    # Original color at the starting pixel
    orig_color = img[y, x].tolist()

    # While there are still pixels to check
    while stack:
        x, y = stack.pop()

        # If the pixel's color matches the original color
        if np.array_equal(img[y, x], orig_color):
            # Change the pixel's color to the new color
            img[y, x] = new_color

            # Add neighboring pixels that are within the image bounds to the stack
            if x > 0:
                stack.append((x - 1, y))
            if x < img.shape[1] - 1:
                stack.append((x + 1, y))
            if y > 0:
                stack.append((x, y - 1))
            if y < img.shape[0] - 1:
                stack.append((x, y + 1))

    return img

def place_colored_pixels(image_path, txt_path):
    """Changes color of specified pixels on an image."""
    # Load the image using OpenCV
    img = cv2.imread(image_path)

    # Read the text file
    with open(txt_path, 'r') as file:
        lines = file.readlines()

    # For each line in the file
    for line in lines:
        # Check if line contains ': '
        if ': ' in line:
            # Split the line on ': ' to separate the key and value
            key, value = line.split(': ')
            
            # Strip the parentheses from the value and split it on ', ' to get the x and y coordinates
            x, y = value.strip('()\n').split(', ')
            
            # Convert the coordinates to integers
            x = int(x)
            y = int(y)

            # Check if the pixel at this location is white (assuming white is [255, 255, 255])
            if np.array_equal(img[y, x], [255, 255, 255]):
                # Change the color to a random warm color using flood fill
                img = flood_fill(img, x, y, random_warm_color())

    # Save the image
    cv2.imwrite('colored_image.png', img)

    # Display the image
    cv2.imshow('Image with colored pixels', img)
    cv2.waitKey(0)  # Wait for the user to press any key
    cv2.destroyAllWindows()

place_colored_pixels("inverted_combined_mask.png", "text_centers.txt")
