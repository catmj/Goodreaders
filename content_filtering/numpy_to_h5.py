import numpy as np
import h5py
import os

# --- Configuration ---
ORIGINAL_NPY_FILEPATH = 'book_similarity_matrix_0-8.npy' 
OUTPUT_H5_FILEPATH = 'book_similarity_matrix.h5'
H5_DATASET_NAME = 'similarity_matrix'

# --- Conversion Logic ---
def convert_npy_to_h5(npy_path, h5_path, dataset_name):
    """
    Loads a NumPy array from an .npy file and saves it as an HDF5 file.
    Includes gzip compression for efficiency.
    """
    if not os.path.exists(npy_path):
        print(f"Error: Original .npy file not found at '{npy_path}'. Please check the path.")
        return

    print(f"Loading large NumPy array from '{npy_path}'...")
    try:
        # Load the entire 8GB array into memory
        # This step will require sufficient RAM on your local machine (at least 8GB + overhead).
        large_array = np.load(npy_path)
        print(f"Successfully loaded array with shape {large_array.shape} and dtype {large_array.dtype}.")
    except Exception as e:
        print(f"Error loading .npy file: {e}")
        return

    print(f"Saving array to HDF5 file '{h5_path}' with dataset '{dataset_name}' and gzip compression...")
    try:
        with h5py.File(h5_path, 'w') as f:
            # Create a dataset within the HDF5 file
            # 'data=large_array' populates it with your NumPy data
            # 'compression="gzip"' applies gzip compression, which is generally good for numerical data.
            # You can experiment with other compressions like "lzf" for speed.
            dset = f.create_dataset(dataset_name, data=large_array, compression="gzip")
        print(f"Successfully converted '{npy_path}' to '{h5_path}'.")
        print(f"HDF5 file size: {os.path.getsize(h5_path) / (1024**3):.2f} GB")
    except Exception as e:
        print(f"Error saving to HDF5 file: {e}")

if __name__ == "__main__":
    convert_npy_to_h5(ORIGINAL_NPY_FILEPATH, OUTPUT_H5_FILEPATH, H5_DATASET_NAME)

