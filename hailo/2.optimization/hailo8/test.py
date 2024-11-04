import numpy as np
import cv2

# Sample inference result from the Hailo hardware (replace this with actual inference results)
inference_output = [
    np.array([[[[1.1539462, -1.1843132, 1.1843132, -1.1539462, 1.1843132, -1.1539462],
                [1.1539462, -1.1843132, 1.1843132, -1.1539462, 1.1843132, -1.1539462],
                [1.1843132, -1.1843132, 1.1843132, -1.1843132, 1.1843132, -1.1843132]]]]),
    np.array([[[[0.37451744, 0.6658088, -1.3316176, 0.70742184, -2.4135568, 0.70742184],
                [0.0, 0.37451744, -1.5812958, 0.41613048, -1.9974264, 0.41613048]]]])
]

def extract_bounding_boxes(output_array):
    """
    Extract bounding boxes from the inference output.
    """
    boxes = output_array.flatten()  # Flatten the array
    # Assuming bounding boxes are stored as [x_min, y_min, x_max, y_max]
    boxes = boxes.reshape((-1, 4))  # Adjust based on how your model outputs boxes
    return boxes

def extract_probabilities(prob_array):
    """
    Extract probabilities (confidence scores) from the inference output.
    """
    probs = prob_array.flatten()
    return probs

def draw_bounding_boxes(image, boxes, probs, threshold=0.5):
    """
    Draw bounding boxes on the image based on the bounding boxes and probabilities.
    """
    for i, box in enumerate(boxes):
        if probs[i] >= threshold:
            x_min, y_min, x_max, y_max = box
            # Draw rectangle for the bounding box
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            # Label the bounding box with the confidence score
            label = f"Conf: {probs[i]:.2f}"
            cv2.putText(image, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Process raw inference output
bounding_boxes = extract_bounding_boxes(inference_output[0])  # Assuming the first array holds bounding boxes
probabilities = extract_probabilities(inference_output[1])    # Assuming the second array holds probabilities

# Load or create an image to draw bounding boxes on (using a placeholder image for now)
image = np.zeros((480, 640, 3), dtype=np.uint8)  # Placeholder black image

# Draw the bounding boxes on the image
output_image = draw_bounding_boxes(image, bounding_boxes, probabilities, threshold=0.5)

# Display the image with bounding boxes
cv2.imshow("Detections", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
