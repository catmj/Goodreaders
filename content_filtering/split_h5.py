import h5py
import numpy as np
import os

# --- Configuration for Splitting ---
LARGE_H5_FILEPATH = 'book_similarity_matrix.h5'
LARGE_H5_DATASET_NAME = 'similarity_matrix'
OUTPUT_DIR = 'split_h5_chunks'
CHUNK_SIZE_ROWS = 3000 # Define how many rows each chunk file should contain. Adjust as needed.

def split_h5_file(input_h5_path, input_dataset_name, output_directory, chunk_size_rows):
    """
    Splits a large HDF5 dataset into multiple smaller HDF5 files.

    Args:
        input_h5_path (str): Path to the large input HDF5 file.
        input_dataset_name (str): Name of the dataset within the input HDF5 file.
        output_directory (str): Directory where the smaller HDF5 chunk files will be saved.
        chunk_size_rows (int): Number of rows to include in each output HDF5 chunk file.
    """
    if not os.path.exists(input_h5_path):
        print(f"Error: Input HDF5 file not found at '{input_h5_path}'.")
        return

    os.makedirs(output_directory, exist_ok=True)
    print(f"Output directory '{output_directory}' ensured.")

    try:
        with h5py.File(input_h5_path, 'r') as f_in:
            if input_dataset_name not in f_in:
                print(f"Error: Dataset '{input_dataset_name}' not found in '{input_h5_path}'.")
                return

            original_dataset = f_in[input_dataset_name]
            total_rows = original_dataset.shape[0]
            print(f"Original dataset shape: {original_dataset.shape}")

            num_chunks = (total_rows + chunk_size_rows - 1) // chunk_size_rows
            print(f"Splitting into {num_chunks} chunks, each with approx {chunk_size_rows} rows.")

            for i in range(num_chunks):
                start_row = i * chunk_size_rows
                end_row = min((i + 1) * chunk_size_rows, total_rows)
                
                chunk_data = original_dataset[start_row:end_row, :] # Read a slice of data

                output_chunk_filepath = os.path.join(output_directory, f"chunk_{i:03d}.h5") # e.g., chunk_000.h5
                output_chunk_dataset_name = f"chunk_data_{i:03d}" # Dataset name inside each chunk file

                with h5py.File(output_chunk_filepath, 'w') as f_out:
                    f_out.create_dataset(output_chunk_dataset_name, data=chunk_data, compression="gzip")
                print(f"Saved chunk {i+1}/{num_chunks} to '{output_chunk_filepath}' (Rows {start_row}-{end_row-1})")

        print("\nHDF5 splitting complete.")

    except Exception as e:
        print(f"An error occurred during splitting: {e}")

# --- Main execution for splitting ---
if __name__ == "__main__":
    split_h5_file(LARGE_H5_FILEPATH, LARGE_H5_DATASET_NAME, OUTPUT_DIR, CHUNK_SIZE_ROWS)
