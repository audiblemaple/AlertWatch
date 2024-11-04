# General imports used throughout the tutorial
# file operations
import json
import os

import numpy as np
import tensorflow as tf
from IPython.display import SVG
from matplotlib import patches
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.python.eager.context import eager_mode

# import the hailo sdk client relevant classes
from hailo_sdk_client import ClientRunner, InferenceContext

IMAGES_TO_VISUALIZE = 5

# Second, we will load our parsed HAR from the Parsing Tutorial

model_name = "face-landmarks-detection"
hailo_model_har_name = f"{model_name}.har"
assert os.path.isfile(hailo_model_har_name), "Please provide valid path for HAR file"
runner = ClientRunner(har=hailo_model_har_name)
# By default it uses the hw_arch that is saved on the HAR. For overriding, use the hw_arch flag.


# Now we will create a model script, that tells the compiler to add a normalization on the beginning
# of the model (that is why we didn't normalize the calibration set;
# Otherwise we would have to normalize it before using it)

# Batch size is 8 by default
alls = "normalization1 = normalization([10], [5])\n"

# Load the model script to ClientRunner so it will be considered on optimization
runner.load_model_script(alls)

calib_dataset = np.load("calib_set.npy")
# Check the shape to ensure it matches the expected format
print(f"Calibration dataset shape: {calib_dataset.shape}")
# Call Optimize to perform the optimization process
runner.optimize(calib_dataset)

# Save the result state to a Quantized HAR file
quantized_model_har_path = f"{model_name}_quantized_model.har"
runner.save_har(quantized_model_har_path)

