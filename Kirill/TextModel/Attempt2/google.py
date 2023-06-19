
def detect_text(path):
    """Detects text in the file and draws bounding boxes around the text."""
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

    for text in texts:
        vertices = [(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices]

        # Draw bounding box
        for i in range(len(vertices)):
            cv2.line(img, vertices[i], vertices[(i + 1) % len(vertices)], (0, 255, 0), 3)

    # Display the image
    cv2.imshow('Image with bounding boxes', img)
    cv2.waitKey(0)  # Wait for the user to press any key
    cv2.destroyAllWindows()

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

detect_text("35.0C original.png")
