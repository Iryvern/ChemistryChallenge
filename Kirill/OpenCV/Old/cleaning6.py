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

def mask_1(img):
    largest_component = find_largest_connected_component(img)
    longest_contour = find_longest_contour(img)

    if longest_contour is not None and largest_component is not None:
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

        # Subtract the yellow mask from the combined_mask
        combined_mask_no_yellow = cv2.subtract(combined_mask, cv2.merge((yellow_mask, yellow_mask, yellow_mask)))

         # Invert the colors of combined_mask_no_yellow
        inverted_combined_mask = cv2.bitwise_not(combined_mask_no_yellow)

        # Save the inverted_combined_mask as a PNG file
        cv2.imwrite("inverted_combined_mask.png", inverted_combined_mask)

        return combined_mask_no_yellow , img_with_mask

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


def show_image(image_path):
   
    img = cv2.imread(image_path)
    
    combined_mask_no_yellow, img_with_mask = mask_1(img)

    # Convert combined_mask_no_yellow to grayscale
    combined_mask_no_yellow_gray = cv2.cvtColor(combined_mask_no_yellow, cv2.COLOR_BGR2GRAY)

    # Concatenate the original image with the combined mask without yellow
    concatenated_img = np.concatenate((img_with_mask, combined_mask_no_yellow), axis=1)

    # Show the concatenated image
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", concatenated_img)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage:
image_path = "40Original.png"
show_image(image_path)
