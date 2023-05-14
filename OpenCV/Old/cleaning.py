import cv2
import pytesseract

def draw_bounding_boxes(image, data):
    for i in range(len(data["level"])):
        confidence = float(data["conf"][i])
        if confidence >= 50:  # Draw box only if confidence is above 50%
            (x, y, w, h) = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            label = f"{confidence}%"
            cv2.putText(image, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            char = data["text"][i]
            print(f"Character: {char} Confidence: {confidence}%")

def show_image(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if img is None:
        print("Error: Image not found or unable to open.")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # OCR: Detect characters and their bounding boxes using Tesseract
    config = "--psm 6"  # PSM 6 is the single block of text mode
    data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)

    # Create a copy of the grayscale image with the bounding box areas covered in white
    white_image = gray.copy()
    for i in range(len(data["level"])):
        (x, y, w, h) = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
        confidence = float(data["conf"][i])
        if confidence >= 50:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(white_image, (x, y), (x + w, y + h), (255, 255, 255), -1)
            label = f"{confidence}%"
            cv2.putText(img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            char = data["text"][i]
            print(f"Character: {char} Confidence: {confidence}%")

    # Show the original image with bounding boxes and the white image side by side
    img_with_boxes = cv2.hconcat([img, cv2.cvtColor(white_image, cv2.COLOR_GRAY2BGR)])
    cv2.namedWindow("Image with bounding boxes and white boxes", cv2.WINDOW_NORMAL)
    cv2.imshow("Image with bounding boxes and white boxes", img_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
image_path = "35Original.png"
show_image(image_path)
