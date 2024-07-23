from hailo_sdk_client import ClientRunner

# Load the quantized HAR model
model_name = 'face_landmarks_trained'
quantized_model_har_path = f'{model_name}_quantized_model.har'
runner = ClientRunner(har=quantized_model_har_path)

# Compile the model
hef = runner.compile()

# Save the compiled HEF file
hef_file_name = f'{model_name}.hef'
with open(hef_file_name, 'wb') as f:
    f.write(hef)

# Save the compiled HAR model
compiled_har_path = f'{model_name}_compiled_model.har'
runner.save_har(compiled_har_path)

# Optionally, run the profiler (assuming you have Hailo profiler installed and configured)
import os
os.system(f'hailo profiler {compiled_har_path}')
