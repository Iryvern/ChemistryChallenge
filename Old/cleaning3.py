import cv2
import numpy as np

# Your existing find_longest_contour and find_largest_connected_component functions

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

def find_largest_connected_component(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    num_labels, labels = cv2.connectedComponents(threshold)

    largest_component = None
    max_area = 0

    for label in range(1, num_labels):
        component_mask = np.where(labels == label, 255, 0).astype('uint8')
        area = cv2.countNonZero(component_mask)
        if area > max_area:
            max_area = area
            largest_component = component_mask

    return largest_component

def create_contour_mask(img, contour):
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [contour], 0, (255, 255, 255), 2)
    return mask

def create_blue_mask(img, component_mask):
    blue_mask = np.zeros_like(img)
    blue_mask[:, :, 0] = component_mask
    return blue_mask

def show_image(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if img is None:
        print("Error: Image not found or unable to open.")
        return

    # Find the longest continuous black line and the largest connected component
    longest_contour = find_longest_contour(img)
    largest_component = find_largest_connected_component(img)

    if longest_contour is not None and largest_component is not None:
        # Create a mask for the longest contour
        contour_mask = create_contour_mask(img, longest_contour)

        # Subtract the longest contour mask from the largest connected component
        remaining_lines = cv2.subtract(largest_component, cv2.cvtColor(contour_mask, cv2.COLOR_BGR2GRAY))

        # Find connected components in the remaining_lines
        num_labels, labels = cv2.connectedComponents(remaining_lines)

        # Create an empty mask for valid remaining lines
        valid_remaining_lines = np.zeros_like(remaining_lines)

        for label in range(1, num_labels):
            component_mask = np.where(labels == label, 255, 0).astype('uint8')
            area = cv2.countNonZero(component_mask)

            # Skip the component if the area is less than 5
            if area < 10 or area > 50:
                continue

            # Draw the valid remaining lines onto the new mask
            valid_remaining_lines = cv2.add(valid_remaining_lines, component_mask)

            # Draw the ID above the line
            x, y, w, h = cv2.boundingRect(component_mask)
            img = cv2.putText(img, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Print the ID and the area
            print(f"ID: {label}, Area: {area}")

        # Create a blue mask for the valid remaining lines
        blue_mask = create_blue_mask(img, valid_remaining_lines)

        # Combine the original image with the blue mask
        img_with_blue_mask = cv2.addWeighted(img, 1, blue_mask, 1, 0)

        # Show the image with the blue mask
        cv2.namedWindow("Image with Blue Mask", cv2.WINDOW_NORMAL)
        cv2.imshow("Image with Blue Mask", img_with_blue_mask)
    else:
        # Show an error message if no line is found
        print("Error: No line found in image.")

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
image_path = "35Original.png"
show_image(image_path)
