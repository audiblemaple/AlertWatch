import numpy as np
import matplotlib.pyplot as plt

# Path to the .npy file
file_path = '../../testing-for-forum/calib_set.npy'

# Load the .npy file
try:
    calib_set = np.load(file_path, allow_pickle=True)
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit()
except Exception as e:
    print(f"An error occurred while loading the file: {e}")
    exit()

# Determine the type of data loaded
if isinstance(calib_set, np.ndarray):
    print(f"Loaded a NumPy array with shape: {calib_set.shape} and dtype: {calib_set.dtype}")

    # If it's an object array (e.g., containing dictionaries)
    if calib_set.dtype == 'object':
        calib_set = calib_set.item()  # Convert to Python object if it's a single object

        if isinstance(calib_set, dict):
            for key, value in calib_set.items():
                print(f"\nKey: {key}")
                print(f"Value: {value}")

                # Optional: Visualize if it's image data
                if isinstance(value, np.ndarray) and value.ndim >= 2:
                    plt.imshow(value, cmap='gray')
                    plt.title(f"{key}")
                    plt.axis('off')
                    plt.show()
        else:
            print("Loaded object is not a dictionary. Inspect it manually.")
    else:
        # If it's numerical data, print summaries
        print("First 5 entries:")
        print(calib_set[:5])

        print("\nStatistics:")
        print(f"Mean: {calib_set.mean()}")
        print(f"Std: {calib_set.std()}")
        print(f"Min: {calib_set.min()}")
        print(f"Max: {calib_set.max()}")

        # Optional: Visualize if it's image-like
        if calib_set.ndim == 2 or calib_set.ndim == 3:
            plt.imshow(calib_set, cmap='gray')
            plt.title('Calibration Data Visualization')
            plt.axis('off')
            plt.show()
else:
    print("Loaded data is not a NumPy array. Inspect it manually.")