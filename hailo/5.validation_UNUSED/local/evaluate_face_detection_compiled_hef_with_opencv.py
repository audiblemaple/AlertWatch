import cv2
import numpy as np
import os
import argparse
import time
from hailo_platform import HEF, VDevice, HailoStreamInterface, ConfigureParams, InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType

class HailoInference:
    def __init__(self, hef_path, output_type='UINT8'):
        self.hef = HEF(hef_path)
        self.target = VDevice()
        self.network_group = self._configure_and_get_network_group()
        self.network_group_params = self.network_group.create_params()
        self.input_vstreams_params, self.output_vstreams_params = self._create_vstream_params(output_type)
        self.input_vstream_info, self.output_vstream_info = self._get_and_print_vstream_info()

    def _configure_and_get_network_group(self):
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        network_group = self.target.configure(self.hef, configure_params)[0]
        return network_group

    def _create_vstream_params(self, output_type):
        input_vstreams_params = InputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=FormatType.UINT8)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(self.network_group, format_type=getattr(FormatType, output_type))
        return input_vstreams_params, output_vstreams_params

    def _get_and_print_vstream_info(self):
        input_vstream_info = self.hef.get_input_vstream_infos()
        output_vstream_info = self.hef.get_output_vstream_infos()
        for layer_info in input_vstream_info:
            print(f'Input layer: {layer_info.name} {layer_info.shape} {layer_info.format.type}')
        for layer_info in output_vstream_info:
            print(f'Output layer: {layer_info.name} {layer_info.shape} {layer_info.format.type}')
        return input_vstream_info, output_vstream_info

    def get_input_shape(self):
        return self.hef.get_input_vstream_infos()[0].shape

    def run(self, input_data):
        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            input_dict = {}
            if isinstance(input_data, dict):
                input_dict = input_data
            elif isinstance(input_data, (list, tuple)):
                for layer_info in self.input_vstream_info:
                    input_dict[layer_info.name] = input_data
            else:
                if len(input_data.shape) == 3:
                    input_data = np.expand_dims(input_data, axis=0)
                input_dict[self.input_vstream_info[0].name] = input_data

            with self.network_group.activate(self.network_group_params):
                output = infer_pipeline.infer(input_dict)[self.output_vstream_info[0].name]
        return output

    def release_device(self):
        self.target.release()


def draw_model_output_on_frame(output, frame):
    """
    Function to draw model output on the frame.
    Args:
    - output: numpy array from the model, expected to be [x1, y1, x2, y2, ...]
    - frame: original video frame to draw on
    """
    # Assuming `output` contains multiple points in a 2D array: [x1, y1, x2, y2, ...]
    # We will iterate over the output and draw each point
    if len(output.shape) > 2:  # If it's a multidimensional array
        output = output.reshape((-1, 2))  # Reshape to (num_points, 2)

    for point in output:
        # Extract the coordinates
        x, y = int(point[0]), int(point[1])

        # Draw the point on the frame as a circle
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Green circles for landmarks

    return frame

def preprocess_image(image, target_size):
    h, w = image.shape[:2]

    # Check for zero height or width and raise an error or skip the frame
    if h == 0 or w == 0:
        raise ValueError("Invalid face ROI dimensions: height or width is zero.")

    scale = min(target_size[0] / h, target_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    dx, dy = (target_size[1] - nw) // 2, (target_size[0] - nh) // 2

    resized_image = cv2.resize(image, (nw, nh))

    # Handle 3-channel images (RGB/BGR)
    if len(image.shape) == 3 and image.shape[2] == 3:
        new_image = np.zeros((target_size[0], target_size[1], 3), np.uint8)  # 3 channels
    else:
        new_image = np.zeros((target_size[0], target_size[1]), np.uint8)  # Grayscale fallback

    new_image[dy:dy + nh, dx:dx + nw] = resized_image
    new_image = np.expand_dims(new_image, axis=0)  # Add batch dimension

    return new_image, scale, dx, dy, nw, nh

def run_ssd_inference(hailo_inference, frame, confidence_threshold=0.5):
    # Get the input shape from the network
    input_height, input_width, _ = hailo_inference.get_input_shape()

    # Preprocess the frame for SSD inference
    preprocessed_frame, scale, dx, dy, nw, nh = preprocess_image(frame, (input_height, input_width))

    # Run SSD inference
    result = hailo_inference.run(preprocessed_frame)

    # Assuming the SSD model output is a numpy array where each detection has:
    # [class_id, score, x_min, y_min, x_max, y_max]
    result = result.reshape((-1, 6))  # Reshape to match the SSD output format

    # Parse the SSD output and extract bounding boxes
    detections = []
    for detection in result:
        score = detection[1]  # Assuming the second value is the confidence score
        if score > confidence_threshold:  # Confidence threshold check
            x_min = int(detection[2] * frame.shape[1])
            y_min = int(detection[3] * frame.shape[0])
            x_max = int(detection[4] * frame.shape[1])
            y_max = int(detection[5] * frame.shape[0])

            # Log bounding box dimensions for debugging
            print(f"Detected bounding box: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}, score={score}")

            # Validate bounding box dimensions
            if x_min < x_max and y_min < y_max:  # Valid bounding box check
                detections.append((x_min, y_min, x_max, y_max))
            else:
                print("Warning: Detected invalid bounding box dimensions.")

    return detections

def main():
    parser = argparse.ArgumentParser(description="Face Detection with Output Drawing using Hailo Platform")
    parser.add_argument("-n", "--net", help="Path for the HEF model.", required=True)
    parser.add_argument("--video_path", help="Path to video file (optional). If not provided, the webcam will be used.", type=str, default=None)
    parser.add_argument("--threshold", default=0.5, type=float, help="Detection confidence threshold.")
    args = parser.parse_args()

    # Load the HEF model
    hailo_inference = HailoInference(args.net, output_type='UINT8')

    # Get input shape for the model
    input_height, input_width, _ = hailo_inference.get_input_shape()

    # Video capture from webcam or video file
    if args.video_path:
        cap = cv2.VideoCapture(args.video_path)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Set resolution to 640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_start_time = 0
    fps = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        frame_count += 1

        # Run SSD inference for face detection
        faces = run_ssd_inference(hailo_inference, frame)

        # Process each detected face
        for (x_min, y_min, x_max, y_max) in faces:
            # Validate bounding box dimensions
            if x_max <= x_min or y_max <= y_min:
                print("Warning: Detected invalid bounding box dimensions.")
                continue  # Skip this invalid face ROI

            # Extract the face ROI
            face_roi = frame[y_min:y_max, x_min:x_max]

            try:
                # Preprocess the face ROI for landmark detection
                preprocessed_face, scale, dx, dy, nw, nh = preprocess_image(face_roi, (input_height, input_width))
            except ValueError as e:
                print(f"Error in preprocessing face ROI: {e}")
                continue  # Skip this face ROI if there's an issue

            # Run inference for landmarks on the face ROI
            result = hailo_inference.run(preprocessed_face)

            # Draw the model's output (e.g., landmarks) on the frame
            frame = draw_model_output_on_frame(result, frame)

        # Calculate FPS
        fps_end_time = time.time()
        time_diff = fps_end_time - fps_start_time
        if time_diff > 0:
            fps = frame_count / time_diff

        # Display FPS on the frame
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (105, 105, 105), 2, cv2.LINE_AA)

        # Display face detection status
        if len(faces) > 0:
            face_detected_text = "Face Detected"
            color = (0, 255, 0)  # Green color
        else:
            face_detected_text = "No Face Detected"
            color = (0, 0, 255)  # Red color

        cv2.putText(frame, face_detected_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Face Detection with Output', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hailo_inference.release_device()
    print("Detection complete")

if __name__ == "__main__":
    main()