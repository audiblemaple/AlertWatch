import cv2
import numpy as np
from hailo_sdk_client import ClientRunner, InferenceContext
import time

# Load the quantized HAR model
model_name = 'face_landmarks_trained'
quantized_har_file = f'{model_name}_quantized_model.har'
runner = ClientRunner(har=quantized_har_file)

# Specify the path to the Haar Cascade file directly
haar_cascade_path = '../../../haarcascades/haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Check if the cascade file loaded correctly
if face_cascade.empty():
    raise IOError(f"Could not load Haar cascade file from {haar_cascade_path}")

# Define the preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, target_size)
    normalized_image = resized_image / 255.0
    expanded_image = np.expand_dims(normalized_image, axis=[0, -1])
    return expanded_image

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

    # Detect faces in the frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        # Preprocess the face ROI
        preprocessed_face = preprocess_image(face_roi)

        # Run inference on the preprocessed face
        with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
            results = runner.infer(ctx, preprocessed_face)
        
        # Process the results
        landmarks = results[0].reshape(-1, 2)
        landmarks = (landmarks * [w, h]).astype(np.int)  # Scale landmarks back to the face ROI size
        

        # Visualize the results
        for (lx, ly) in landmarks:
            cv2.circle(frame, (x + lx, y + ly), 2, (0, 255, 0), -1)
    
    # Display the FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with landmarks
    cv2.imshow('Facial Landmarks', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
