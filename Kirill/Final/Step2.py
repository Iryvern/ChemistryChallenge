import os
os.environ["GOOGLE_CLOUD_PROJECT"] = "adroit-coral-389910"

def detect_text(path):
    """Detects text in the file and draws circles at the center of the bounding boxes."""
    from google.cloud import vision
    import cv2
    import numpy as np

    client = vision.ImageAnnotatorClient()

    with open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    # Load the image using OpenCV
    img = cv2.imread(path)

    # Initialize an empty dictionary to store the detected texts and their corresponding center coordinates
    text_centers = {}

    for text in texts:
        vertices = [(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices]

        # Calculate the center of the bounding box
        center_x = int(sum(vertex[0] for vertex in vertices) / len(vertices))
        center_y = int(sum(vertex[1] for vertex in vertices) / len(vertices))

        # Draw a red circle at the center of the bounding box
        cv2.circle(img, (center_x, center_y), radius=2, color=(0, 0, 255), thickness=-1)

        # Write the coordinates on the image
        cv2.putText(img, f"({center_x}, {center_y})", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Add the detected text and its center coordinates to the dictionary
        text_centers[text.description] = (center_x, center_y)

    # Display the image
    cv2.imshow('Image with circles', img)
    cv2.waitKey(0)  # Wait for the user to press any key
    cv2.destroyAllWindows()

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    # Clear the text file before writing to it
    with open('text_centers.txt', 'w') as file:
        file.write('')

    # Write the dictionary of detected texts and their center coordinates to the text file
    with open('text_centers.txt', 'a') as file:
        for key, value in text_centers.items():
            file.write(f'{key}: {value}\n')

    print(text_centers)

detect_text("OutputSmaller.png")
