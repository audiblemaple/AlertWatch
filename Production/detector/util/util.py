"""
File System Utilities

This module provides utility functions for file system operations.

Functions:
    - ensure_directory_exists: Ensures a directory exists, creating it if necessary.

Author:
    Lior Jigalo

License:
    MIT
"""

import os

"""
Create a directory if it does not already exist.

Args:
    directory (str): The path of the directory to create.
"""
def ensure_directory_exists(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")