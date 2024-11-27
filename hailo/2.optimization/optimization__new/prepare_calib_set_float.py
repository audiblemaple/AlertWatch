import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


# Define preprocessing steps using torchvision.transforms
preprocess = transforms.Compose([
    transforms.Resize(256),  # Aspect-preserving resize (smaller side = 256)
    transforms.CenterCrop(224),  # Central crop to 224x224
    transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor and scale [0, 255] to [0, 1]
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])


def preprocess_image_pil(image, transform):
    """
    Preprocess a PIL Image using the specified transform.

    Parameters:
    - image (PIL.Image): The input image to preprocess.
    - transform (torchvision.transforms.Compose): The preprocessing pipeline.

    Returns:
    - np.ndarray: The preprocessed image as a NumPy array.
    """
    # Apply the transformations
    tensor = transform(image)  # Shape: (1, 224, 224)

    tensor = tensor.permute(1, 2, 0)  # Shape: (224, 224, 1)

    # Convert tensor to NumPy array
    image_np = tensor.numpy()  # Shape: (1, 224, 224)

    return image_np


def create_calibration_set(images_path, output_file, transform):
    """
    Create a calibration set by preprocessing images and saving them as a .npy file.

    Parameters:
    - images_path (str): Path to the directory containing face images.
    - output_file (str): Filename for the output .npy file.
    - transform (torchvision.transforms.Compose): The preprocessing pipeline.

    Returns:
    - None
    """
    # Get a list of image files with .png, .jpg, or .jpeg extensions
    images_list = [img_name for img_name in os.listdir(images_path)
                   if os.path.splitext(img_name)[1].lower() in [".png", ".jpg", ".jpeg"]]

    # Initialize a list to store preprocessed images
    calib_dataset = []

    for idx, img_name in enumerate(sorted(images_list)):
        img_path = os.path.join(images_path, img_name)

        # Open the image and convert to grayscale
        img = Image.open(img_path).convert("L")  # "L" mode ensures it's grayscale

        # Preprocess the image
        img_preproc = preprocess_image_pil(img, transform)  # NumPy array shape: (1, 224, 224)

        # Append to the dataset
        calib_dataset.append(img_preproc)  # Shape: (1, 224, 224)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(images_list):
            print(f"Processed {idx + 1}/{len(images_list)} images.")

    # Convert the list to a NumPy array
    calib_dataset = np.array(calib_dataset, dtype=np.float32)  # Shape: (num_images, 1, 224, 224)

    # Save the calibration dataset as a .npy file
    np.save(output_file, calib_dataset)
    print(f"Calibration dataset saved to {output_file}")


if __name__ == "__main__":
    # Define input directory and output file
    images_path = "processed_faces"  # Replace with your actual input directory containing cropped face images
    output_file = "calib_set.npy"

    # Create the calibration set
    create_calibration_set(images_path, output_file, preprocess)
