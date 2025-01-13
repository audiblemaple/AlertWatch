import os
import numpy as np
import tensorflow as tf
from PIL import Image


# Preprocess function to resize and crop the image
def preproc(image, output_height=224, output_width=224, resize_side=256):
    """imagenet-standard: aspect-preserving resize to 256px smaller-side, then central-crop to 224px"""
    h, w = image.shape[0], image.shape[1]
    scale = tf.cond(tf.less(h, w), lambda: resize_side / h, lambda: resize_side / w)
    # Expand dims to add batch and channel dimensions
    resized_image = tf.compat.v1.image.resize_bilinear(tf.expand_dims(tf.expand_dims(image, -1), 0),
                                                       [int(h * scale), int(w * scale)])
    cropped_image = tf.compat.v1.image.resize_with_crop_or_pad(resized_image, output_height, output_width)

    return tf.squeeze(cropped_image)


# Define input directory and output file
images_path = "processed_faces"  # Replace with your actual input directory containing cropped face images
output_file = "calib_set.npy"

# Get a list of image files
images_list = [img_name for img_name in os.listdir(images_path) if os.path.splitext(img_name)[1].lower() == ".png"]

# Initialize an empty array to store the calibration data
calib_dataset = np.zeros((len(images_list), 224, 224, 1), dtype=np.uint8)

for idx, img_name in enumerate(sorted(images_list)):
    # Open the grayscale image
    img_path = os.path.join(images_path, img_name)
    img = np.array(Image.open(img_path).convert("L"))  # Ensure it is grayscale

    # Preprocess the image
    img_preproc = preproc(img)

    # Add the processed image to the dataset
    calib_dataset[idx, :, :, 0] = img_preproc.numpy()

# Save the calibration dataset as a .npy file
np.save(output_file, calib_dataset)
print(f"Calibration dataset saved to {output_file}")
