import cv2
import numpy as np
from hailo_sdk_client import ClientRunner, InferenceContext
import time
import torch

# Load the quantized HAR model
model_name = 'output/face-detection-RFB-640-optimized.har'  # Path to your Hailo .har model
quantized_har_file = f'{model_name}'
runner = ClientRunner(har=quantized_har_file)


# Define the preprocessing function
def preprocess_image(image, target_size=(640, 480)):
    resized_image = cv2.resize(image, target_size)
    normalized_image = resized_image / 255.0
    expanded_image = np.expand_dims(normalized_image, axis=0)  # Shape: (1, 480, 640, 3)
    return expanded_image


# Define NMS (Non-Maximal Suppression) to filter out redundant bounding boxes
def nms(boxes, scores, iou_threshold=0.3):
    picked_boxes = []
    picked_scores = []

    indices = np.argsort(scores)[::-1]  # Sort scores in descending order

    while len(indices) > 0:
        current = indices[0]
        picked_boxes.append(boxes[current])
        picked_scores.append(scores[current])

        if len(indices) == 1:
            break

        current_box = boxes[current]
        remaining_boxes = boxes[indices[1:]]

        # Compute IoU
        ious = compute_iou(current_box, remaining_boxes)
        indices = indices[1:][ious <= iou_threshold]

    return np.array(picked_boxes), np.array(picked_scores)


def compute_iou(box1, boxes):
    x1 = np.maximum(box1[0], boxes[:, 0])
    y1 = np.maximum(box1[1], boxes[:, 1])
    x2 = np.minimum(box1[2], boxes[:, 2])
    y2 = np.minimum(box1[3], boxes[:, 3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    return inter_area / (box1_area + boxes_area - inter_area)


# Capture video from webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set resolution to 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fps_start_time = 0
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps = 1 / time_diff
    fps_start_time = fps_end_time

    # Preprocess the frame for inference (resize to 640x480, no grayscale conversion)
    preprocessed_frame = preprocess_image(frame)

    # Run inference on the preprocessed frame using Hailo SDK
    with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
        results = runner.infer(ctx, preprocessed_frame)

        print("boxes: ")
        boxes = torch.from_numpy(results[0])
        # print(boxes)
        print("type boxes: ", type(boxes))

        print("scores: ")
        scores = torch.from_numpy(results[1])
        # print(scores)
        print("type scores: ", type(scores))
        print("scores size: ", scores.size(1))






        scores = torch.from_numpy(results[2])
        # print(scores)
        print("type scores: ", type(scores))
        print("scores size: ", scores.size(1))


        scores = torch.from_numpy(results[3])
        # print(scores)
        print("type scores: ", type(scores))
        print("scores size: ", scores.size(1))


        scores = torch.from_numpy(results[4])
        # print(scores)
        print("type scores: ", type(scores))
        print("scores size: ", scores.size(1))

        scores = torch.from_numpy(results[5])
        # print(scores)
        print("type scores: ", type(scores))
        print("scores size: ", scores.size(1))

        scores = torch.from_numpy(results[6])
        # print(scores)
        print("type scores: ", type(scores))
        print("scores size: ", scores.size(1))

        scores = torch.from_numpy(results[7])
        # print(scores)
        print("type scores: ", type(scores))
        print("scores size: ", scores.size(1))




        # print("probs: ")
        # print(results[2])
        #
        # print("result[3]: ",results[3])
        # print("result[4]: ",results[4])
        # print("result[5]: ",results[5])
        # print("result[6]: ",results[6])
        # print("result[7]: ",results[7])
        # print("result[8]: ",results[8])



        exit(1)

    # Assuming the model outputs bounding boxes and probabilities (modify based on your actual output)
    boxes, scores = results[0], results[0]  # Adjust based on actual output format

    # Apply NMS to get filtered bounding boxes and corresponding scores
    picked_boxes, picked_scores = nms(boxes.reshape(-1, 4), scores.flatten(), iou_threshold=0.3)

    # Visualize the results
    for box in picked_boxes:
        # The box is in normalized coordinates, so we need to convert it to pixel coordinates
        x1_norm, y1_norm, x2_norm, y2_norm = box

        # The frame's width and height
        height, width, _ = frame.shape

        print("norm: ", x1_norm, y1_norm, x2_norm, y2_norm)

        # Convert normalized coordinates to pixel values
        x1 = int((x1_norm ) / 2 * width)
        y1 = int((y1_norm ) / 2 * (-height))
        x2 = int((x2_norm ) / 2 * width)
        y2 = int((y2_norm ) / 2 * (-height))

        # Clip coordinates to ensure they are within the frame bounds
        # x1 = max(0, min(x1, width))
        # y1 = max(0, min(y1, height))
        # x2 = max(0, min(x2, width))
        # y2 = max(0, min(y2, height))

        print("reg: ", x1, y1, x2, y2, "\n")  # Print the clipped pixel coordinates

        # Draw the bounding box on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow('Face Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
