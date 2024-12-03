from hailo_platform import HEF, VDevice, HailoStreamInterface, ConfigureParams, InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType
import numpy as np

class HailoInference_async:
    def __init__(self, hef_path, input_type='UINT8', output_type='UINT8'):
        self.input_type = input_type
        self.output_type = output_type
        self.hef = HEF(hef_path)
        self.target = VDevice()
        self._configure_network_group()

    def _configure_network_group(self):
        """Configure the network group and vstreams."""
        self.network_group = self._configure_and_get_network_group()
        self.network_group_params = self.network_group.create_params()
        self.input_vstreams_params, self.output_vstreams_params = self._create_vstream_params(self.input_type, self.output_type)
        self.input_vstream_info, self.output_vstream_info = self._get_vstream_info()

    def _configure_and_get_network_group(self):
        """Configure the network group using the HEF."""
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        network_group = self.target.configure(self.hef, configure_params)[0]
        return network_group

    def _create_vstream_params(self, input_type, output_type):
        """Create vstream parameters for input and output."""
        input_vstreams_params = InputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=getattr(FormatType, input_type)
        )
        output_vstreams_params = OutputVStreamParams.make_from_network_group(
            self.network_group, format_type=getattr(FormatType, output_type)
        )
        return input_vstreams_params, output_vstreams_params

    def _get_vstream_info(self):
        """Get vstream information for input and output."""
        input_vstream_info = self.hef.get_input_vstream_infos()
        output_vstream_info = self.hef.get_output_vstream_infos()
        return input_vstream_info, output_vstream_info

    def get_input_shape(self):
        """Return the shape of the input layer."""
        return self.hef.get_input_vstream_infos()[0].shape

    def run(self, input_data):
        """Run inference with the configured network group."""
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
                output = infer_pipeline.infer(input_dict)
        return output

    def release_device(self):
        """Release the VDevice."""
        self.target.release()
