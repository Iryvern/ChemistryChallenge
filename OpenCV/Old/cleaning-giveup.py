import cv2
import numpy as np

# This function takes an input image and finds the longest contour in it
def find_longest_contour(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply a threshold to convert the image to a binary image with white pixels representing the object
    _, threshold = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Find the contours in the binary image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store the longest contour and its length
    longest_contour = None
    max_length = 0

    # Iterate over all the contours found in the image
    for contour in contours:
        # Calculate the length of the current contour
        length = cv2.arcLength(contour, True)
        # If the length of the current contour is greater than the maximum length seen so far,
        # update the maximum length and longest contour variables
        if length > max_length:
            max_length = length
            longest_contour = contour

    # Return the longest contour found in the image
    return longest_contour

# This function takes an input image and finds the largest connected component in it
def find_largest_connected_component(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply a threshold to convert the image to a binary image with white pixels representing the object
    _, threshold = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Find the connected components in the binary image
    num_labels, labels = cv2.connectedComponents(threshold)

    # Initialize variables to store the largest connected component and its area
    largest_component = None
    max_area = 0

    # Iterate over all the connected components found in the image
    for label in range(1, num_labels):
        # Create a binary mask for the current connected component
        component_mask = np.where(labels == label, 255, 0).astype('uint8')
        # Calculate the area of the current connected component
        area = cv2.countNonZero(component_mask)
        # If the area of the current connected component is greater than the maximum area seen so far,
        # update the maximum area and largest connected component variables
        if area > max_area:
            max_area = area
            largest_component = component_mask

    # Return the largest connected component found in the image as a binary mask
    return largest_component

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

def isolate_triangle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    largest_triangle = None
    largest_area = 0

    mask = np.zeros_like(img)  # Initialize the mask with the same dimensions as the input image

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
                mask = np.zeros_like(img)  # Reset the mask
                cv2.drawContours(mask, [polygon], -1, (255, 255, 255), -1)  # Draw the largest triangle on the mask

    return mask

def create_mask(img, component_mask):
    mask = np.zeros_like(img)
    mask[:, :, 2] = component_mask
    return mask

def create_black_mask(img, component_mask):
    black_mask = np.zeros_like(img)
    black_mask[:, :, 0] = component_mask
    black_mask[:, :, 1] = component_mask
    black_mask[:, :, 2] = component_mask
    return black_mask

def create_blue_mask(img, component_mask):
    blue_mask = np.zeros_like(img)
    blue_mask[:, :, 0] = component_mask
    return blue_mask

def create_contour_mask(img, contour):
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [contour], 0, (255, 255, 255), 2)
    return mask

def show_image(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if img is None:
        print("Error: Image not found or unable to open.")
        return

    # Find the largest connected component
    largest_component = find_largest_connected_component(img)
    longest_contour = find_longest_contour(img)

    if longest_contour is not None and largest_component is not None:
        # Create a mask with the largest connected component
        mask = create_mask(img, largest_component)
        contour_mask = create_contour_mask(img, longest_contour)
        black_mask = create_black_mask(img, largest_component)

        remaining_lines = cv2.subtract(largest_component, cv2.cvtColor(contour_mask, cv2.COLOR_BGR2GRAY))
        num_labels, labels = cv2.connectedComponents(remaining_lines)
        valid_remaining_lines = np.zeros_like(remaining_lines)

        for label in range(1, num_labels):
            component_mask = np.where(labels == label, 255, 0).astype('uint8')
            area = cv2.countNonZero(component_mask)

            # Skip the component if the area is less than 5
            if area < 15 or area > 60:
                continue

            # Draw the valid remaining lines onto the new mask
            valid_remaining_lines = cv2.add(valid_remaining_lines, component_mask)
        
        # Combine the original image with the mask
        img_with_mask = cv2.addWeighted(img, 1, mask, 1, 0)

        blue_mask = create_blue_mask(img, valid_remaining_lines)
        img_with_blue_mask = cv2.addWeighted(img, 1, blue_mask, 1, 0)

        # Subtract the blue mask from the black mask
        combined_mask = cv2.subtract(black_mask, blue_mask)

        # Create a mask for yellow pixels
        yellow_mask = cv2.inRange(combined_mask, (0, 255, 255), (0, 255, 255))

        #Outer Triangle clear
        triangle_mask = isolate_triangle(img)

        # Subtract the yellow mask from the combined_mask
        combined_mask_no_yellow = cv2.subtract(combined_mask, cv2.merge((yellow_mask, yellow_mask, yellow_mask)))
    
        print("combined_mask_no_yellow shape:", combined_mask_no_yellow.shape)
        print("triangle_mask shape:", triangle_mask.shape)
        print("combined_mask_no_yellow dtype:", combined_mask_no_yellow.dtype)
        print("triangle_mask dtype:", triangle_mask.dtype)
        print("combined_mask_no_yellow min:", np.min(combined_mask_no_yellow))
        print("combined_mask_no_yellow max:", np.max(combined_mask_no_yellow))
        print("triangle_mask min:", np.min(triangle_mask))
        print("triangle_mask max:", np.max(triangle_mask))


            
        #Add the tirangle mask
        combined_mask_no_y_no_out = cv2.subtract(combined_mask_no_yellow, cv2.merge((triangle_mask, triangle_mask, triangle_mask)))

        # Invert the colors of combined_mask_no_yellow
        inverted_combined_mask = cv2.bitwise_not(combined_mask_no_y_no_out)

        # Save the inverted_combined_mask as a PNG file
        cv2.imwrite("inverted_combined_mask.png", inverted_combined_mask)

        # Concatenate the original image with the combined mask without yellow
        concatenated_img = np.concatenate((img_with_mask, combined_mask_no_y_no_out), axis=1)

        # Show the concatenated image
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", concatenated_img)
    else:
        # Show the original image if no line is found
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
image_path = "40Original.png"
show_image(image_path)
