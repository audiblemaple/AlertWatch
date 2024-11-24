import cv2
import argparse
import time

from inference import HailoInference

def main():
    parser = argparse.ArgumentParser(description="Facial Landmarks Detection Example")
    parser.add_argument("-n", "--net", help="Path for the HEF model.", default="model/face-landmarks-detection.hef", required=False)
    parser.add_argument("-a", "--arch", help="Hailo architecture, h8, h15h", required=False)  # For future
    args = parser.parse_args()

    # Load the HEF model
    hailo_inference = HailoInference(args.net, input_type="UINT8", output_type='UINT8')

    # Get input shape
    input_height, input_width, _ = hailo_inference.get_input_shape()

    # Capture video from webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set resolution to 640x480 (or higher if needed for accuracy)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_start_time = 0
    fps = 0

    haar_cascade_path = '../../haarcascades/haarcascade_frontalface_alt.xml'
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    if face_cascade.empty():
        raise IOError(f"Could not load Haar cascade file from {haar_cascade_path}")

    padding = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        fps_end_time = time.time()
        time_diff = fps_end_time - fps_start_time
        fps = 1 / time_diff
        fps_start_time = fps_end_time

        # Run face detection without scaling
        faces = face_cascade.detectMultiScale(frame, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Add padding to the face region
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(frame.shape[1], x + w + padding)
            y_end = min(frame.shape[0], y + h + padding)

            face_roi = frame[y_start:y_end, x_start:x_end]

            # Preprocess the face region
            preprocessed_face = preprocess_face_landmarks(face_roi, target_size=(input_height, input_width))

            # Run inference
            results = hailo_inference.run(preprocessed_face)
            results = results['face-landmarks-detection/fc1']

            # Scale and display landmarks
            landmarks = results[0].reshape(-1, 2)

            # Draw detected face region and landmarks
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            for (lx, ly) in landmarks:
                cv2.circle(frame, (x + lx, y + ly), 2, (0, 255, 0), -1)

        # Display FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2, cv2.LINE_AA)

        # Display face detection status
        face_detected_text = "Face Detected" if len(faces) > 0 else "No Face Detected"
        color = (0, 255, 0) if len(faces) > 0 else (0, 0, 255)
        cv2.putText(frame, face_detected_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Show the frame with landmarks
        cv2.imshow('Webcam Face Landmarks', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    hailo_inference.release_device()

if __name__ == "__main__":
    main()
