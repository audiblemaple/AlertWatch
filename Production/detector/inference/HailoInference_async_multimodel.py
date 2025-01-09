"""
Hailo Inference Module

This module provides functionality to perform asynchronous inference using the Hailo platform.
It supports running multiple models concurrently, configuring network groups, and handling inference pipelines.

Classes:
    - HailoInferenceAsyncMultiModel: Manages multiple inference models on a Hailo device.

Functions:
    - initialize_inference_models: Initializes face detection and face landmarks models for inference.

Dependencies:
    - hailo_platform: For Hailo device and model configuration.
    - numpy: For numerical operations and input data manipulation.
    - subprocess: For interacting with external commands (e.g., `hailortcli`).

Usage:
    Use the `initialize_inference_models` function to set up models for face detection and landmarks inference.

Author:
    Lior Jigalo

License:
    MIT
"""

from hailo_platform import HEF, VDevice, HailoStreamInterface, ConfigureParams, InferVStreams, InputVStreamParams, \
    OutputVStreamParams, FormatType
import numpy as np
import subprocess

class HailoInferenceAsyncMultiModel:
    """
    Manages multiple inference models on a Hailo device.

    This class handles the configuration of two models, their network groups, input/output stream parameters,
    and provides methods to run inference on the models.

    Attributes:
        model_paths (tuple): Paths to the HEF files for the two models.
        models (dict): Contains model-specific configuration and parameters.
        target (VDevice): Represents the Hailo VDevice.

    Methods:
        __init__: Initializes the inference object with two models.
        _get_hailo_architecture: Detects the Hailo device architecture.
        _load_model: Configures a model and sets up its network group.
        get_input_shape: Returns the shape of the input layer for a specific model.
        run: Executes inference on the specified model.
        release_device: Releases the Hailo VDevice.
    """
    def __init__(self, model_paths, input_types=('UINT8', 'UINT8'), output_types=('UINT8', 'UINT8')):
        """
        Initialize the Hailo inference object with two models.

        Args:
            model_paths (tuple): Paths to the two HEF files.
            input_types (tuple): Input types for the two models (default: UINT8).
            output_types (tuple): Output types for the two models (default: UINT8).
        """
        self.model_paths = model_paths
        self.models = {}
        self.target = VDevice()

        for i, (hef_path, input_type, output_type) in enumerate(zip(model_paths, input_types, output_types)):
            self.models[f'model_{i + 1}'] = self._load_model(hef_path, input_type, output_type)

    def _get_hailo_architecture(self):
        """
        Detect the Hailo device architecture by running the `hailortcli` command.

        Returns:
            HailoStreamInterface | str: The appropriate Hailo stream interface or an error message.
        """
        try:
            # Run the hailortcli command and capture the output
            result = subprocess.run(
                ["hailortcli", "fw-control", "identify"],
                capture_output=True,
                text=True
            )

            # Check if the command executed successfully
            if result.returncode != 0:
                print(f"Error executing command: {result.stderr}")
                return None

            # Parse the output to find the architecture
            for line in result.stdout.splitlines():
                if "Device Architecture:" in line:
                    architecture = line.split(":")[1].strip()
                    if architecture == "HAILO8":
                        return HailoStreamInterface.PCIe
                    elif architecture == "HAILO15H":
                        return HailoStreamInterface.INTEGRATED

            return "Device Architecture not found in output."
        except Exception as e:
            return f"An error occurred: {e}"

    def _load_model(self, hef_path, input_type, output_type):
        """
        Load a model and configure its network group.

        Args:
            hef_path (str): Path to the HEF file for the model.
            input_type (str): The input type (e.g., 'UINT8').
            output_type (str): The output type (e.g., 'UINT8').

        Returns:
            dict: Model-specific configuration and parameters.
        """
        hef = HEF(hef_path)

        architecture = self._get_hailo_architecture()

        configure_params = ConfigureParams.create_from_hef(hef, interface=architecture)
        network_group = self.target.configure(hef, configure_params)[0]

        # Create vstream parameters
        input_vstreams_params = InputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=getattr(FormatType, input_type)
        )
        output_vstreams_params = OutputVStreamParams.make_from_network_group(
            network_group, format_type=getattr(FormatType, output_type)
        )

        # Get input and output stream info
        input_vstream_info = hef.get_input_vstream_infos()
        output_vstream_info = hef.get_output_vstream_infos()

        return {
            "hef": hef,
            "network_group": network_group,
            "input_vstreams_params": input_vstreams_params,
            "output_vstreams_params": output_vstreams_params,
            "input_vstream_info": input_vstream_info,
            "output_vstream_info": output_vstream_info,
            "params": network_group.create_params(),
        }

    def get_input_shape(self, model_id=1):
        """
        Return the shape of the input layer for a specific model.

        Args:
            model_id (int): The ID of the model (1 or 2).

        Returns:
            tuple: Shape of the input layer.
        """
        return self.models[f'model_{model_id}']['input_vstream_info'][0].shape

    def run(self, model_id, input_data):
        """
        Run inference on the specified model.

        Args:
            model_id (int): The ID of the model (1 or 2).
            input_data (np.ndarray): The input data for inference.

        Returns:
            dict: Output data from the inference.
        """
        model = self.models[f'model_{model_id}']
        with InferVStreams(model['network_group'], model['input_vstreams_params'],
                           model['output_vstreams_params']) as infer_pipeline:
            input_dict = {}
            if len(input_data.shape) == 3:
                input_data = np.expand_dims(input_data, axis=0)
            input_dict[model['input_vstream_info'][0].name] = input_data

            with model['network_group'].activate(model['params']):
                output = infer_pipeline.infer(input_dict)
        return output

    def release_device(self):
        """
        Release the VDevice.
        """
        self.target.release()

def initialize_inference_models(face_det_path, face_landmark_path):
    """
    Initializes the Hailo inference models for face detection and face landmarks.

    Args:
        face_det_path (str): Path to the face detection HEF file.
        face_landmark_path (str): Path to the face landmarks HEF file.

    Returns:
        tuple: Contains the following:
            - hailo_inference (HailoInferenceAsyncMultiModel): Initialized inference object.
            - face_detection_input_shape (tuple): Input shape for the face detection model.
            - face_landmarks_input_shape (tuple): Input shape for the face landmarks model.
            - face_land_output_name (str): Name of the output stream for the landmarks model.
    """
    model_paths = (face_det_path, face_landmark_path)
    hailo_inference = HailoInferenceAsyncMultiModel(
        model_paths, input_types=('FLOAT32', 'UINT8'), output_types=('FLOAT32', 'FLOAT32')
    )

    face_detection_input_shape = hailo_inference.get_input_shape(model_id=1)
    face_landmarks_input_shape = hailo_inference.get_input_shape(model_id=2)

    face_land_output_name = hailo_inference.models['model_2']['output_vstream_info'][0].name

    return hailo_inference, face_detection_input_shape, face_landmarks_input_shape, face_land_output_name