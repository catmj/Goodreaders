import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from tqdm import tqdm # For progress bars
import os # For path manipulation and directory creation
import json # For potentially saving metadata alongside checkpoints (not fully used here, but good practice)

# --- Define file paths ---
books_with_vectors_filepath = '../keyword_processing/books_with_vectors.csv'
base_output_filename = 'book_similarity_matrix'
# Directory to save intermediate checkpoint files and final outputs
checkpoint_dir = './similarity_checkpoints'

def calculate_book_similarity(books_with_vectors_filepath, genre_weight=0.5, keyword_weight=0.5, checkpoint_dir=None):
    """
    Reads book data with vectors, calculates combined keyword and genre cosine similarity
    matrix, and returns the matrix along with combined book identifiers.
    Includes checkpointing and progress indicators.

    Args:
        books_with_vectors_filepath (str): Path to the CSV file containing book titles, authors,
                                           and keyword/genre vectors as string representations of lists.
        genre_weight (float): Weight for genre similarity.
        keyword_weight (float): Weight for keyword similarity.
        checkpoint_dir (str, optional): Directory for saving/loading intermediate checkpoints.
                                        If None, no checkpointing is performed.

    Returns:
        tuple: A tuple containing:
               - combined_sim_matrix (np.array): The square similarity matrix.
               - combined_headers (list): List of strings "title, author name" for each book.
               - df (pd.DataFrame): The loaded DataFrame for potential further use.
               Returns (None, None, None) if an error occurs or data is insufficient.
    """
    print(f"\n--- Calculating Book Similarity (Genre Weight: {genre_weight}, Keyword Weight: {keyword_weight}) ---")
    print(f"Reading data from: {books_with_vectors_filepath}")

    # Ensure the checkpoint directory exists if checkpointing is enabled
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True) # exist_ok=True prevents error if dir exists
        print(f"Using checkpoint directory: {checkpoint_dir}")

    # Create a unique identifier for this specific (genre_weight, keyword_weight) combination
    # Replace '.' with '-' for safe filename use to avoid issues with file systems
    unique_id = f"g{str(genre_weight).replace('.', '-')}_k{str(keyword_weight).replace('.', '-')}"
    keyword_sim_checkpoint_path = os.path.join(checkpoint_dir, f"keyword_sim_matrix_{unique_id}.npy") if checkpoint_dir else None
    genre_sim_checkpoint_path = os.path.join(checkpoint_dir, f"genre_sim_matrix_{unique_id}.npy") if checkpoint_dir else None

    df = None
    try:
        df = pd.read_csv(books_with_vectors_filepath)
    except FileNotFoundError:
        print(f"Error: The input file '{books_with_vectors_filepath}' was not found.")
        return None, None, None
    except Exception as e:
        print(f"Error reading '{books_with_vectors_filepath}': {e}")
        return None, None, None

    if df.empty:
        print("Input CSV is empty. No similarity matrix to calculate.")
        return None, None, None

    required_cols = ['title', 'author_name', 'keyword_vector', 'genre_vector']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing one or more required columns ({required_cols}) in the CSV.")
        print("Please ensure the input CSV was generated by `export_vectors_to_csv` function.")
        return None, None, None

    keyword_vectors = None
    genre_vectors = None
    keyword_sim_matrix = None
    genre_sim_matrix = None

    # --- Check and load from checkpoints for keyword similarity ---
    if keyword_sim_checkpoint_path and os.path.exists(keyword_sim_checkpoint_path):
        print(f"Attempting to load keyword similarity matrix from checkpoint: '{keyword_sim_checkpoint_path}'...")
        try:
            keyword_sim_matrix = np.load(keyword_sim_checkpoint_path)
            print("Successfully loaded keyword similarity matrix from checkpoint.")
        except Exception as e:
            print(f"Error loading keyword similarity checkpoint: {e}. Will recalculate.")

    # --- Check and load from checkpoints for genre similarity ---
    if genre_sim_checkpoint_path and os.path.exists(genre_sim_checkpoint_path):
        print(f"Attempting to load genre similarity matrix from checkpoint: '{genre_sim_checkpoint_path}'...")
        try:
            genre_sim_matrix = np.load(genre_sim_checkpoint_path)
            print("Successfully loaded genre similarity matrix from checkpoint.")
        except Exception as e:
            print(f"Error loading genre similarity checkpoint: {e}. Will recalculate.")

    # If either matrix wasn't loaded from a checkpoint, proceed with vector conversion and calculation
    if keyword_sim_matrix is None or genre_sim_matrix is None:
        print("Converting vector strings to lists (this may take a moment)...")
        try:
            # Using tqdm to show progress while evaluating string representations of lists
            keyword_vectors = np.array([eval(x) for x in tqdm(df['keyword_vector'], desc="Processing keyword vectors")])
            genre_vectors = np.array([eval(x) for x in tqdm(df['genre_vector'], desc="Processing genre vectors")])
        except Exception as e:
            print(f"Error converting vector strings to lists. Ensure they are valid list representations: {e}")
            return None, None, None

        if keyword_vectors.shape[0] == 0 or genre_vectors.shape[0] == 0:
            print("No valid vectors found after conversion. Cannot calculate similarity.")
            return None, None, None

        # --- Calculate and save keyword similarity if not loaded ---
        if keyword_sim_matrix is None:
            print("Calculating keyword similarity matrix...")
            # cosine_similarity is highly optimized (often C-implemented),
            # so granular progress bars directly inside it are not feasible.
            keyword_sim_matrix = cosine_similarity(keyword_vectors)
            print("Keyword similarity calculation complete.")
            if keyword_sim_checkpoint_path:
                try:
                    np.save(keyword_sim_checkpoint_path, keyword_sim_matrix)
                    print(f"Saved keyword similarity checkpoint to: '{keyword_sim_checkpoint_path}'")
                except Exception as e:
                    print(f"Error saving keyword similarity checkpoint: {e}")

        # --- Calculate and save genre similarity if not loaded ---
        if genre_sim_matrix is None:
            print("Calculating genre similarity matrix...")
            genre_sim_matrix = cosine_similarity(genre_vectors)
            print("Genre similarity calculation complete.")
            if genre_sim_checkpoint_path:
                try:
                    np.save(genre_sim_checkpoint_path, genre_sim_matrix)
                    print(f"Saved genre similarity checkpoint to: '{genre_sim_checkpoint_path}'")
                except Exception as e:
                    print(f"Error saving genre similarity checkpoint: {e}")

    print("Combining similarities based on weights...")
    combined_sim_matrix = (genre_weight * genre_sim_matrix) + (keyword_weight * keyword_sim_matrix)
    print("Combined similarity matrix created.")

    # Prepare headers for the output files
    combined_headers = [
        f"{title}, {author_name}"
        for title, author_name in zip(df['title'], df['author_name'])
    ]

    return combined_sim_matrix, combined_headers, df


def export_similarity_data(combined_sim_matrix, combined_headers, base_output_filename='book_similarity_matrix', g=0.8):
    """
    Exports a similarity matrix to a NumPy .npy file and a text file
    containing the corresponding book identifiers.
    Includes progress indicators for file saving.
    """
    if combined_sim_matrix is None or combined_headers is None:
        print("No similarity matrix or headers provided for export. Skipping export.")
        return

    # Create a unique identifier for the output files based on genre weight
    # Replace '.' with '-' for safe filename use
    output_suffix = str(g).replace('.', '-')

    # --- Export to NumPy .npy file ---
    npy_filepath = f"{base_output_filename}_{output_suffix}.npy"
    print(f"Exporting similarity matrix to NumPy .npy: '{npy_filepath}'...")
    try:
        # np.save is generally efficient, so a simple message suffices.
        np.save(npy_filepath, combined_sim_matrix)
        print(f"Successfully exported NumPy array to '{npy_filepath}'")
    except Exception as e:
        print(f"Error exporting to NumPy .npy: {e}")

    # --- Export list of book identifiers to .txt file ---
    txt_filepath = f"{base_output_filename}_books_{output_suffix}.txt"
    print(f"Exporting book identifiers to text file: '{txt_filepath}'...")
    try:
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            # Using tqdm for progress bar during writing each header line
            for header in tqdm(combined_headers, desc="Writing book headers to .txt"):
                f.write(f"{header}\n")
        print(f"Successfully exported book identifiers to '{txt_filepath}'")
    except Exception as e:
        print(f"Error exporting book identifiers to text file: {e}")

# --- Main execution loop ---
if __name__ == "__main__":
    # Ensure the main checkpoint directory exists at the start of the script
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created main checkpoint directory: {checkpoint_dir}")
    else:
        print(f"Using existing main checkpoint directory: {checkpoint_dir}")

    # Define the different weight pairs to iterate through
    weight_pairs = [(0.8, 0.2)] # [(0.5, 0.5), (0.8, 0.2), (0, 1)]

    for g, k in weight_pairs:
        # Print separators for better readability between runs
        print("\n" + "="*70)
        print(f"STARTING PROCESS FOR Genre Weight = {g}, Keyword Weight = {k}")
        print("="*70)

        # Calculate similarity, passing the checkpoint directory
        combined_sim_matrix, combined_headers, _ = calculate_book_similarity(
            books_with_vectors_filepath,
            genre_weight=g,
            keyword_weight=k,
            checkpoint_dir=checkpoint_dir # Pass the checkpoint directory for saving/loading
        )

        # Export data only if the similarity calculation was successful
        if combined_sim_matrix is not None and combined_headers is not None:
            export_similarity_data(combined_sim_matrix, combined_headers, base_output_filename=base_output_filename, g=g)
        else:
            print(f"Skipping export for (genre_weight={g}, keyword_weight={k}) due to errors during similarity calculation.")

    print("\n" + "="*70)
    print("ALL BOOK SIMILARITY CALCULATIONS COMPLETE.")
    print("="*70)

