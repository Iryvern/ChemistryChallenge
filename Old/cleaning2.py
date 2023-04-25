import cv2
import numpy as np

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

def show_image(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if img is None:
        print("Error: Image not found or unable to open.")
        return

    # Find the largest connected component
    largest_component = find_largest_connected_component(img)

    if largest_component is not None:
        # Create a mask with the largest connected component
        mask = create_mask(img, largest_component)
        black_mask = create_black_mask(img, largest_component)

        # Combine the original image with the mask
        img_with_mask = cv2.addWeighted(img, 1, mask, 1, 0)

        # Concatenate the original image with the black mask
        concatenated_img = np.concatenate((img_with_mask, black_mask), axis=1)

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
image_path = "35Original.png"
show_image(image_path)
