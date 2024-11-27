import torch
import torch.nn as nn
from torchvision import models

# Define your Network model based on ResNet18 directly within this script
class Network(nn.Module):
    """
    A class to define a neural network model based on ResNet-18 architecture,
    customized for landmark detection tasks.
    """
    def __init__(self, num_classes=None):
        """
        Initializes the Network class with the specified number of output classes.
        Args:
            num_classes (int, optional): The number of output classes.
                For eyes, it should be 24 (12 points * 2 for x, y).
                For face, it should be 136 (68 points * 2 for x, y).
        """
        super().__init__()
        self.model_name = 'resnet18'
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Single-channel input (grayscale)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

# Specify the number of output classes (adjust based on your task)
num_classes = 136  # For 68 facial landmarks, 2 coordinates each (x, y)

# Instantiate your model
model = Network(num_classes=num_classes)

# Load the trained .pth file (replace 'path_to_your_model.pth' with the actual path)
model_path = 'TODO: add a static path later'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Set the model to evaluation mode
model.eval()

# Create a dummy input with the same shape that the model expects
dummy_input = torch.randn(1, 1, 224, 224)  # Adjust input size if needed

# Export the model to ONNX format
onnx_output_path = "model.onnx"
torch.onnx.export(
    model,                      # Model to be exported
    dummy_input,                # Example input tensor
    onnx_output_path,           # The file where the ONNX model will be saved
    export_params=True,         # Store the trained parameters
    opset_version=11,           # ONNX version to export to
    do_constant_folding=True,   # Whether to execute constant folding for optimization
    input_names=['input'],      # Input node name
    output_names=['output'],    # Output node name
    dynamic_axes={
        'input': {0: 'batch_size'},    # Enable dynamic batching for input
        'output': {0: 'batch_size'}    # Enable dynamic batching for output
    }
)

print(f"Model successfully exported to ONNX format at {onnx_output_path}")
import torch
import torch.nn as nn
from torchvision import models

# Define your Network model based on ResNet18 directly within this script
class Network(nn.Module):
    """
    A class to define a neural network model based on ResNet-18 architecture,
    customized for landmark detection tasks.
    """
    def __init__(self, num_classes=None):
        """
        Initializes the Network class with the specified number of output classes.
        Args:
            num_classes (int, optional): The number of output classes.
                For eyes, it should be 24 (12 points * 2 for x, y).
                For face, it should be 136 (68 points * 2 for x, y).
        """
        super().__init__()
        self.model_name = 'resnet18'
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Single-channel input (grayscale)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

# Specify the number of output classes (adjust based on your task)
num_classes = 136  # For 68 facial landmarks, 2 coordinates each (x, y)

# Instantiate your model
model = Network(num_classes=num_classes)

# Load the trained .pth file (replace 'path_to_your_model.pth' with the actual path)
model_path = '../../hailo/1.parsing_Final/models/face_landmarks_epoch_9319.pth'  # Adjust to your model path
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Set the model to evaluation mode
model.eval()

# Create a dummy input with the same shape that the model expects
dummy_input = torch.randn(1, 1, 224, 224)  # Adjust input size if needed

# Export the model to ONNX format
onnx_output_path = "model.onnx"
torch.onnx.export(
    model,                      # Model to be exported
    dummy_input,                # Example input tensor
    onnx_output_path,           # The file where the ONNX model will be saved
    export_params=True,         # Store the trained parameters
    opset_version=11,           # ONNX version to export to
    do_constant_folding=True,   # Whether to execute constant folding for optimization
    input_names=['input'],      # Input node name
    output_names=['output'],    # Output node name
    dynamic_axes={
        'input': {0: 'batch_size'},    # Enable dynamic batching for input
        'output': {0: 'batch_size'}    # Enable dynamic batching for output
    }
)

print(f"Model successfully exported to ONNX format at {onnx_output_path}")
