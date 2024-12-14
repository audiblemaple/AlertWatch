from hailo_platform import HEF, VDevice, HailoStreamInterface, ConfigureParams, InferVStreams, InputVStreamParams, \
    OutputVStreamParams, FormatType
import numpy as np
import subprocess

class HailoInferenceAsyncMultiModel:
    def __init__(self, model_paths, input_types=('UINT8', 'UINT8'), output_types=('UINT8', 'UINT8')):
        """
        Initialize the Hailo inference object with two models.

        Args:
            model_paths (tuple): Paths to the two HEF files.
            input_types (tuple): Input types for the two models.
            output_types (tuple): Output types for the two models.
        """
        self.model_paths = model_paths
        self.models = {}
        self.target = VDevice()

        for i, (hef_path, input_type, output_type) in enumerate(zip(model_paths, input_types, output_types)):
            self.models[f'model_{i + 1}'] = self._load_model(hef_path, input_type, output_type)

    def _get_hailo_architecture(self):
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
        """Return the shape of the input layer for a specific model."""
        return self.models[f'model_{model_id}']['input_vstream_info'][0].shape

    def run(self, model_id, input_data):
        """
        Run inference on the specified model.

        Args:
            model_id (int): The ID of the model (1 or 2).
            input_data (np.ndarray): The input data for inference.
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
        """Release the VDevice."""
        self.target.release()
