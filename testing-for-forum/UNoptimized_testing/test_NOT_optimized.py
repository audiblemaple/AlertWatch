# Import necessary modules
from hailo_sdk_client.runner.client_runner import ClientRunner
from hailo_sdk_client.exposed_definitions import InferenceContext
import numpy as np
import cv2

# Path to your optimized HAR file
har_path = 'face-landmarks-detection.har'


# Load the HAR file into the ClientRunner
runner = ClientRunner(har=har_path)

# Check the runner's state
print(f"Runner state after loading HAR: {runner.state}")

# Perform full-precision optimization to prepare for inference
runner.optimize_full_precision()

# Use the SDK_NATIVE inference context to run inference on the unoptimized model
inference_context = InferenceContext.SDK_NATIVE

# Open a connection to the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Set the camera resolution to 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Load Haar Cascade for face detection
# You may need to download 'haarcascade_frontalface_default.xml' from OpenCV's GitHub repository
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Prepare for inference
with runner.infer_context(inference_context) as ctx:
    while True:
        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame using Haar Cascades
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        # Process each face detected
        for (x, y, w, h) in faces:
            # Crop the face region
            face_roi = gray[y:y+h, x:x+w]

            # Resize the face region to the required input size (e.g., 224x224)
            input_size = (224, 224)  # Adjust based on your model's expected input size
            face_resized = cv2.resize(face_roi, input_size)

            # Normalize the image if required (e.g., scale pixel values to [0, 1])
            face_normalized = face_resized / 255.0

            # Add batch dimension and channel dimension to match the model's expected input shape
            # Expected input shape: [batch_size, height, width, channels]
            # Since the model uses grayscale images, channels = 1
            input_data = np.expand_dims(face_normalized, axis=(0, -1))  # Shape: (1, 224, 224, 1)

            # Perform inference using the runner
            outputs = runner.infer(ctx, input_data, data_type='np_array')

            # Process the output
            landmarks = outputs[0]  # Assuming the model has a single output

            # Reshape the landmarks to get pairs of (x, y) coordinates
            # The output shape is (batch_size, 136), where 136 = 68 landmarks * 2 coordinates (x and y)
            # We reshape it to (68, 2)
            landmarks = np.reshape(landmarks, (-1, 2))

            # The coordinates might be normalized between 0 and 1 or in pixel coordinates
            # Adjust scaling based on your model's output
            # If normalized, scale them back to the size of the face ROI
            # If already in pixel coordinates, you may not need scaling
            landmarks[:, 0] *= w  # Scale x coordinates if needed
            landmarks[:, 1] *= h  # Scale y coordinates if needed

            # Shift the landmarks to the position of the face ROI in the original frame
            landmarks[:, 0] += x
            landmarks[:, 1] += y

            # Draw the landmarks on the original frame
            for (lx, ly) in landmarks.astype(np.int32):
                cv2.circle(frame, (lx, ly), 2, (0, 255, 0), -1)

            # Optionally, draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Face Landmarks Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
