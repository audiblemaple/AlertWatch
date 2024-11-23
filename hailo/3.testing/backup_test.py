import cv2
import numpy as np
import logging
from multiprocessing import Process
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, FormatType)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HailoInference():
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
        input_vstreams_params = InputVStreamParams.make_from_network_group(self.network_group, format_type=self.hef.get_input_vstream_infos()[0].format.type)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(self.network_group, format_type=getattr(FormatType, output_type))
        return input_vstreams_params, output_vstreams_params

    def _get_and_print_vstream_info(self):
        input_vstream_info = self.hef.get_input_vstream_infos()
        output_vstream_info = self.hef.get_output_vstream_infos()

        for layer_info in input_vstream_info:
            logger.info('Input layer: {} {} {}'.format(layer_info.name, layer_info.shape, layer_info.format.type))
        for layer_info in output_vstream_info:
            logger.info('Output layer: {} {} {}'.format(layer_info.name, layer_info.shape, layer_info.format.type))

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


def preprocess_frame(frame, height, width):
    frame_resized = cv2.resize(frame, (width, height))
    frame_normalized = frame_resized / 255.0
    frame_normalized = np.expand_dims(frame_normalized, axis=0)
    frame_normalized = np.expand_dims(frame_normalized, axis=0)
    return frame_normalized.astype(np.uint8)  # Change dtype to uint8


def main():
    hef_path = 'face-landmarks-detection.hef'
    haarcascade_path = 'haarcascades/haarcascade_frontalface_alt.xml'
    hailo_inference = HailoInference(hef_path)
    height, width, _ = hailo_inference.get_input_shape()

    face_cascade = cv2.CascadeClassifier(haarcascade_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(grayscale_image, 1.1, 4)

        all_landmarks = []
        for (x, y, w, h) in faces:
            face = grayscale_image[y:y + h, x:x + w]
            face = preprocess_frame(face, height, width)

            infer_result = hailo_inference.run(face)
            infer_result = infer_result.reshape(-1, 2)

            landmarks = (infer_result * np.array([[w, h]])) + np.array([[x, y]])
            all_landmarks.append(landmarks)

        for landmarks in all_landmarks:
            for (x, y) in landmarks:
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

        cv2.imshow('Webcam Face Landmarks', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hailo_inference.release_device()


if __name__ == "__main__":
    main()
