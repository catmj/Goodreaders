import pandas as pd
import numpy as np
import h5py
import os

# --- Define file paths ---
# BASE_FILE_NAME is now primarily for the book identifiers TXT file
BASE_FILE_NAME = '../content_filtering/book_similarity_matrix' 
GENRE_WEIGHT = 0.8 

# Constants for the HDF5 chunks
H5_CHUNKS_DIR = '../content_filtering/split_h5_chunks' 
H5_CHUNK_FILE_PREFIX = 'chunk_' # Prefix for chunk filenames (e.g., chunk_000.h5)
H5_CHUNK_DATASET_PREFIX = 'chunk_data_' # Prefix for dataset names inside chunk files (e.g., chunk_data_000)
CHUNK_SIZE_ROWS = 3000 # IMPORTANT: This must match the chunking size used during file splitting.

BOOK_IDENTIFIERS_TXT_FILEPATH = f"{BASE_FILE_NAME}_books_{str(GENRE_WEIGHT).replace('.', '-')}.txt"

# Cache for loaded HDF5 chunk files to avoid reopening the same file repeatedly
# This will hold h5py.File objects. Keys will be filepaths.
_H5_FILE_CACHE = {}

def _get_h5_file_handle(filepath):
    """
    Returns an HDF5 file handle from a cache or opens a new one.
    This helps reduce overhead if the same chunk file is accessed multiple times.
    """
    if filepath not in _H5_FILE_CACHE:
        try:
            _H5_FILE_CACHE[filepath] = h5py.File(filepath, 'r')
            print(f"Opened HDF5 file: {filepath}")
        except Exception as e:
            print(f"Error opening HDF5 file {filepath}: {e}")
            return None
    return _H5_FILE_CACHE[filepath]

def _close_all_h5_file_handles():
    """Closes all cached HDF5 file handles."""
    for filepath, f_handle in list(_H5_FILE_CACHE.items()): 
        try:
            f_handle.close()
            print(f"Closed HDF5 file: {filepath}")
        except Exception as e:
            print(f"Error closing HDF5 file {filepath}: {e}")
        del _H5_FILE_CACHE[filepath] 

def _load_book_identifiers(book_identifiers_txt_filepath):
    """
    Loads book identifiers from a text file.
    """
    if not os.path.exists(book_identifiers_txt_filepath):
        print(f"Error: Book identifiers file not found at '{book_identifiers_txt_filepath}'.")
        return None
    try:
        with open(book_identifiers_txt_filepath, 'r', encoding='utf-8') as f:
            all_book_identifiers = [line.strip() for line in f]
        print(f"Successfully loaded all book identifiers from '{book_identifiers_txt_filepath}' ({len(all_book_identifiers)} entries)")
        return all_book_identifiers
    except Exception as e:
        print(f"Error loading book identifiers: {e}")
        return None

def _normalize_ratings(user_ratings):
    """
    Normalizes user ratings to a 0-1 scale based on the user's own min/max rating.
    If all ratings are identical, they are treated as a positive weight of 1.0.
    """
    user_ratings_np = np.array(user_ratings)

    normalized_ratings = []

    # Check if all user ratings are identical
    if np.all(user_ratings_np == user_ratings_np[0]):
        normalized_ratings = [1.0] * len(user_ratings)
    else:
        user_min_rating = user_ratings_np.min()
        user_max_rating = user_ratings_np.max()
        user_rating_range = user_max_rating - user_min_rating

        if user_rating_range == 0: 
            normalized_ratings = [1] * len(user_ratings) 
            print(f"Warning: User's rating range is zero despite having varying ratings, normalized to 1.")
        else:
            for rating_val in user_ratings:
                normalized_ratings.append((rating_val - user_min_rating) / user_rating_range)
    
    normalized_ratings_np = np.array(normalized_ratings)
 
    return normalized_ratings_np

def _map_user_books_to_indices(user_books, user_ratings, all_book_identifiers):
    """
    Maps user's rated books to their corresponding indices in the similarity matrix
    and associates them with their original ratings.
    """
    user_book_indices = []
    user_book_original_ratings = []
    
    for i, book_id in enumerate(user_books):
        try:
            idx_in_matrix = all_book_identifiers.index(book_id)
            user_book_indices.append(idx_in_matrix)
            user_book_original_ratings.append(user_ratings[i])
        except ValueError:
            print(f"Warning: Your rated book '{book_id}' not found in the overall book list. Skipping.")

    if not user_book_indices:
        print("No rated books from your list were found in the similarity matrix. Cannot generate recommendations.")
        return None, None
    
    return np.array(user_book_indices), np.array(user_book_original_ratings)

def _calculate_combined_weighted_scores(user_book_indices_np, normalized_user_ratings_np, total_books_count):
    """
    Calculates the combined weighted similarity score for all books.
    This function now directly accesses the HDF5 chunks as needed, avoiding full matrix load.
    """
    combined_weighted_similarity_scores = np.zeros(total_books_count, dtype=np.float32)
    sum_abs_weights = np.sum(np.abs(normalized_user_ratings_np))

    if sum_abs_weights == 0:
        print("All your found rated books resulted in zero-sum normalized ratings. Cannot generate meaningful weighted recommendations.")
        return None

    # Group user book indices by the chunk they belong to
    chunks_to_load = {} # {chunk_index: [relative_idx_1, relative_idx_2, ...]}
    for user_idx in user_book_indices_np:
        chunk_idx = user_idx // CHUNK_SIZE_ROWS
        relative_idx_in_chunk = user_idx % CHUNK_SIZE_ROWS
        if chunk_idx not in chunks_to_load:
            chunks_to_load[chunk_idx] = []
        chunks_to_load[chunk_idx].append(relative_idx_in_chunk)

    # Iterate through the chunks that contain user-rated books
    for chunk_idx in sorted(chunks_to_load.keys()):
        chunk_filepath = os.path.join(H5_CHUNKS_DIR, f"{H5_CHUNK_FILE_PREFIX}{chunk_idx:03d}.h5")
        chunk_dataset_name = f"{H5_CHUNK_DATASET_PREFIX}{chunk_idx:03d}"

        f_chunk = _get_h5_file_handle(chunk_filepath)
        if f_chunk is None:
            print(f"Warning: Could not get file handle for chunk {chunk_idx}. Skipping.")
            continue
        
        if chunk_dataset_name not in f_chunk:
            print(f"Warning: Dataset '{chunk_dataset_name}' not found in chunk file '{chunk_filepath}'. Skipping this chunk.")
            continue

        chunk_dataset = f_chunk[chunk_dataset_name]

        # Get the global indices for this chunk that correspond to user books
        relevant_user_indices_in_chunk_global = [
            global_idx for global_idx in user_book_indices_np
            if chunk_idx == (global_idx // CHUNK_SIZE_ROWS)
        ]
        
        
        for i, user_idx_global in enumerate(user_book_indices_np):
            if chunk_idx == (user_idx_global // CHUNK_SIZE_ROWS):
                rating_weight = normalized_user_ratings_np[i]
                relative_idx_in_chunk = user_idx_global % CHUNK_SIZE_ROWS
                
                try:
                    # Read only the specific row needed from the current chunk
                    similarities_from_this_book = chunk_dataset[relative_idx_in_chunk, :]
                    combined_weighted_similarity_scores += similarities_from_this_book * rating_weight
                except Exception as e:
                    print(f"Error reading row {relative_idx_in_chunk} from chunk {chunk_idx}: {e}. Skipping this row.")
                    continue
    
    combined_weighted_similarity_scores /= sum_abs_weights
    print("Combined weighted similarity scores calculated by accessing chunks.")
    return combined_weighted_similarity_scores

def _prepare_full_recommendation_list(combined_weighted_scores, all_book_identifiers, user_books, user_ratings):
    """
    Prepares a DataFrame including all books with their calculated similarity scores,
    marking user-rated books and including their original ratings.
    """
    all_books_data = []
    
    # Create a dictionary for quick lookup of user's rated books and their ratings
    user_rated_books_dict = {user_books[i]: user_ratings[i] for i in range(len(user_books))}

    for i, book_id in enumerate(all_book_identifiers):
        score = combined_weighted_scores[i]
        is_rated_by_user = book_id in user_rated_books_dict
        original_rating = user_rated_books_dict.get(book_id) if is_rated_by_user else np.nan

        all_books_data.append({
            'Book Identifier': book_id,
            'Weighted Similarity Score': score,
            'Rated by User': is_rated_by_user,
            'Original Rating': original_rating
        })

    result_df = pd.DataFrame(all_books_data)
    result_df = result_df.sort_values(by='Weighted Similarity Score', ascending=False).reset_index(drop=True)
    
    print(f"Successfully compiled list of {len(result_df)} books including user-rated books.")
    return result_df

def find_all_books_with_scores(book_identifiers_txt_filepath, user_books, user_ratings):
    """
    Finds similarity scores for all books based on a user's list of rated books,
    including the books the user has rated themselves. The output includes a column
    indicating if the book was rated by the user and their original rating.

    Args:
        book_identifiers_txt_filepath (str): Path to the text file containing the book identifiers
                                             (one per line), corresponding to the HDF5 matrix order.
        user_books (list of str): A list of book identifiers (e.g., "Title by Author Name") that the user has rated.
        user_ratings (list of int): A list of corresponding ratings for `user_books` (e.g., 1-5 scale).

    Returns:
        pd.DataFrame: A DataFrame with 'Book Identifier', 'Weighted Similarity Score',
                      'Rated by User', and 'Original Rating' columns, sorted by weighted
                      similarity in descending order. Returns an empty DataFrame
                      if no valid rated books are found or an error occurs during loading.
    """
    print(f"\n--- Finding All Books with Similarity Scores (including user-rated books) ---")

    if not user_books or not user_ratings or len(user_books) != len(user_ratings):
        print("Error: 'user_books' and 'user_ratings' must be non-empty lists of the same length.")
        return pd.DataFrame()

    # 1. Load book identifiers
    all_book_identifiers = _load_book_identifiers(book_identifiers_txt_filepath)
    if all_book_identifiers is None:
        return pd.DataFrame()

    # 2. Normalize Ratings (used for calculating combined scores)
    normalized_user_ratings_np = _normalize_ratings(user_ratings)

    # 3. Map User Books to Matrix Indices (and get original ratings for output)
    user_book_indices_np, _ = _map_user_books_to_indices(
        user_books, user_ratings, all_book_identifiers
    )
    if user_book_indices_np is None: # Means no user books found in the matrix
        return pd.DataFrame()

    # 4. Calculate Combined Weighted Similarity Scores directly from chunks
    combined_weighted_scores = _calculate_combined_weighted_scores(
        user_book_indices_np, normalized_user_ratings_np, len(all_book_identifiers)
    )
    if combined_weighted_scores is None:
        return pd.DataFrame()

    # 5. Prepare the full list including rated books
    full_recommendation_df = _prepare_full_recommendation_list(
        combined_weighted_scores, all_book_identifiers, user_books, user_ratings
    )
    
    _close_all_h5_file_handles()

    return full_recommendation_df
