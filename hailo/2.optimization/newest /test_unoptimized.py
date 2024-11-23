import cv2
import numpy as np
from hailo_sdk_client import ClientRunner, InferenceContext
import os

# HAR model path
hailo_model_har_name = "face-landmarks-detection.har"
assert os.path.isfile(hailo_model_har_name), "Please provide valid path for HAR file"
runner = ClientRunner(har=hailo_model_har_name)

# Preprocessing function
def preprocess_image(image, target_size=(224, 224), normalize=True):
    """
    Preprocesses an image for inference:
    - Resize to the target size
    - Convert to grayscale
    - Normalize the pixel values if required
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32)
    if normalize:
        image = (image / 255.0 - 0.5) / 0.5  # Normalize to [-1, 1]
    return image

# Postprocessing function
def postprocess_landmarks(output, bbox):
    """
    Scales the predicted landmarks back to the original image dimensions.
    """
    x, y, w, h = bbox
    landmarks = (output + 0.5) * np.array([w, h]) + np.array([x, y])
    return landmarks

# Draw landmarks on the frame
def draw_landmarks(image, landmarks):
    for (x, y) in landmarks:
        cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)
    return image

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)  # Use webcam (change index if necessary)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting inference with OpenCV feed...")

# Real-time inference loop
with runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from webcam.")
            break

        orig_frame = frame.copy()

        # Define a dummy bounding box (use face detection to replace this if available)
        bbox = [0, 0, frame.shape[1], frame.shape[0]]  # Whole frame as bbox

        # Preprocess frame
        preprocessed_frame = preprocess_image(frame)
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=(0, -1))  # Add batch and channel dimensions

        # Run inference
        native_res = runner.infer(ctx, preprocessed_frame)

        # Debugging: Print the structure of the inference result
        print(f"native_res type: {type(native_res)}")
        print(f"native_res content: {native_res}")

        # Process landmarks based on the structure of native_res
        if isinstance(native_res, dict):
            # If output is a dictionary, extract the specific key
            landmarks = native_res["face-landmarks-detection/fc1"].reshape(-1, 2)
        elif isinstance(native_res, np.ndarray):
            # If output is a NumPy array, use it directly
            landmarks = native_res.reshape(-1, 2)
        else:
            raise TypeError(f"Unexpected type for native_res: {type(native_res)}")

        # Postprocess landmarks
        scaled_landmarks = postprocess_landmarks(landmarks, bbox)

        # Draw landmarks on the original frame
        output_frame = draw_landmarks(orig_frame, scaled_landmarks)

        # Display the frame
        cv2.imshow("Real-Time Face Landmarks", output_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
