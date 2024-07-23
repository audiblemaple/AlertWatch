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
            print('Input layer: {} {} {}'.format(layer_info.name, layer_info.shape, layer_info.format.type))
        for layer_info in output_vstream_info:
            print('Output layer: {} {} {}'.format(layer_info.name, layer_info.shape, layer_info.format.type))
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

def preprocess_image(image, target_size):
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    dx, dy = (target_size[1] - nw) // 2, (target_size[0] - nh) // 2

    resized_image = cv2.resize(image, (nw, nh))
    new_image = np.zeros((target_size[0], target_size[1]), np.uint8)
    new_image[dy:dy+nh, dx:dx+nw] = resized_image
    new_image = np.expand_dims(new_image, axis=-1)  # Add channel dimension
    new_image = np.expand_dims(new_image, axis=0)   # Add batch dimension

    return new_image, scale, dx, dy, nw, nh

def main():
    parser = argparse.ArgumentParser(description="Facial Landmarks Detection Example")
    parser.add_argument("-n", "--net", help="Path for the HEF model.", required=True)
    args = parser.parse_args()
    
    # Load the HEF model
    hailo_inference = HailoInference(args.net, output_type='UINT8')
    
    # Get input shape
    input_height, input_width, _ = hailo_inference.get_input_shape()
    
    # Capture video from webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set resolution to 640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_start_time = 0
    fps = 0

    haar_cascade_path = '../haarcascades/haarcascade_frontalface_alt.xml'
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    if face_cascade.empty():
        raise IOError(f"Could not load Haar cascade file from {haar_cascade_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        fps_end_time = time.time()
        time_diff = fps_end_time - fps_start_time
        fps = 1 / time_diff
        fps_start_time = fps_end_time

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray_frame.shape

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        all_landmarks = []
        for (x, y, w, h) in faces:
            # Extract the face ROI
            face_roi = gray_frame[y:y+h, x:x+w]
            
            # Preprocess the face ROI
            preprocessed_face, scale, dx, dy, nw, nh = preprocess_image(face_roi, (input_height, input_width))

            # Run inference on the preprocessed face
            result = hailo_inference.run(preprocessed_face)
            landmarks = result.reshape(-1, 2)

            # Scale and translate landmarks back to the original image coordinates
            landmarks[:, 0] = (landmarks[:, 0] - dx) / scale * w / nw + x
            landmarks[:, 1] = (landmarks[:, 1] - dy) / scale * h / nh + y

            all_landmarks.append(landmarks)

            # Draw the detected face region
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Draw the preprocessed face region
            cv2.rectangle(frame, (x + dx, y + dy), (x + dx + nw, y + dy + nh), (0, 255, 0), 2)

        print("new frame...")
        # Visualize the results
        for landmarks in all_landmarks:
            for (lx, ly) in landmarks:
                print((int(lx), int(ly)))
                cv2.circle(frame, (int(lx), int(ly)), 2, (0, 0, 255), -1)

        # Display the FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2, cv2.LINE_AA)  # Dark grey color

        # Display face detection status
        if len(faces) > 0:
            face_detected_text = "Face Detected"
            color = (0, 255, 0)  # Green color
        else:
            face_detected_text = "No Face Detected"
            color = (0, 0, 255)  # Red color

        cv2.putText(frame, face_detected_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        cv2.imshow('Webcam Face Landmarks', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hailo_inference.release_device()
    print("Done")

if __name__ == "__main__":
    main()
