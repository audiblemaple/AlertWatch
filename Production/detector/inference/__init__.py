"""
Hailo Inference Module Initialization

This package provides classes and functions for performing inference on Hailo devices.
The `__init__.py` file initializes the package and exposes key classes and methods.

Modules:
    - HailoInference: Provides synchronous inference functionality.
    - HailoInference_async: Provides asynchronous inference functionality.
    - HailoInference_async_multimodel: Supports inference with multiple models on a Hailo device.

Exports:
    - HailoInference: Class for managing synchronous inference.
    - HailoInference_async: Class for managing asynchronous inference.
    - HailoInferenceAsyncMultiModel: Class for managing multiple inference models asynchronously.
    - initialize_inference_models: Function to initialize models for inference.

Usage:
    Import the desired class or function from the package for inference on Hailo devices.
"""
from .HailoInference import HailoInference
from .HailoInference_async import HailoInference_async
from .HailoInference_async_multimodel import HailoInferenceAsyncMultiModel, initialize_inference_models