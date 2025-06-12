import pandas as pd
import numpy as np

# --- Define file paths ---
BASE_FILE_NAME = '../content_filtering/book_similarity_matrix' 
GENRE_WEIGHT = 0.8
SIMILARITY_MATRIX_NPY_FILEPATH = f"{BASE_FILE_NAME}_{str(GENRE_WEIGHT).replace('.', '-')}.npy"
BOOK_IDENTIFIERS_TXT_FILEPATH = f"{BASE_FILE_NAME}_books_{str(GENRE_WEIGHT).replace('.', '-')}.txt"

def _load_similarity_data(similarity_matrix_npy_filepath, book_identifiers_txt_filepath):
    """
    Loads the similarity matrix and corresponding book identifiers from files.
    """
    try:
        sim_matrix = np.load(similarity_matrix_npy_filepath)
        print(f"Successfully loaded similarity matrix from '{similarity_matrix_npy_filepath}' (Shape: {sim_matrix.shape})")

        with open(book_identifiers_txt_filepath, 'r', encoding='utf-8') as f:
            all_book_identifiers = [line.strip() for line in f]
        print(f"Successfully loaded all book identifiers from '{book_identifiers_txt_filepath}' ({len(all_book_identifiers)} entries)")
        return sim_matrix, all_book_identifiers
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}.")
        return None, None
    except Exception as e:
        print(f"Error loading similarity data: {e}")
        return None, None

def _normalize_ratings(user_ratings):
    """
    Normalizes user ratings to a 0-1 scale based on the user's own min/max rating.
    If all ratings are identical, they are treated as a positive weight of 1.0.
    """
    user_ratings_np = np.array(user_ratings)
    print(f"User's raw ratings: {user_ratings}")

    normalized_ratings = []

    # Check if all user ratings are identical
    if np.all(user_ratings_np == user_ratings_np[0]):
        # If all original ratings are identical (e.g., all 3s, all 5s, all 1s),
        # treat them as a positive weight of 1.0 for recommendations.
        normalized_ratings = [1.0] * len(user_ratings)
        print(f"All your ratings are identical, treating them as a positive weight of 1 for recommendations.")
    else:
        user_min_rating = user_ratings_np.min()
        user_max_rating = user_ratings_np.max()
        
        # Calculate the effective range of the user's ratings
        user_rating_range = user_max_rating - user_min_rating

        if user_rating_range == 0: 
            # This case should ideally be caught by the `np.all` condition
            normalized_ratings = [1] * len(user_ratings) # Fallback to neutral if range is unexpectedly zero
            print(f"Warning: User's rating range is zero despite having varying ratings, normalized to 1. This should have been caught by the 'all identical' check.")
        else:
            # Min-Max scaling using user's own min and max
            # Formula: (x - user_min) / (user_max - user_min)
            for rating_val in user_ratings:
                normalized_ratings.append((rating_val - user_min_rating) / user_rating_range)
    
    normalized_ratings_np = np.array(normalized_ratings)
    print(f"Normalized (0-1 scaled based on user's range) ratings: {normalized_ratings_np}")
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

def _calculate_combined_weighted_scores(sim_matrix, user_book_indices_np, normalized_user_ratings_np, total_books_count):
    """
    Calculates the combined weighted similarity score for all books in the matrix
    based on the user's rated books and their weights.
    """
    combined_weighted_similarity_scores = np.zeros(total_books_count)
    sum_abs_weights = np.sum(np.abs(normalized_user_ratings_np))

    if sum_abs_weights == 0:
        print("All your found rated books resulted in zero-sum normalized ratings. Cannot generate meaningful weighted recommendations.")
        return None

    for i, user_idx in enumerate(user_book_indices_np):
        rating_weight = normalized_user_ratings_np[i]
        similarities_from_this_book = sim_matrix[user_idx, :]
        combined_weighted_similarity_scores += similarities_from_this_book * rating_weight

    combined_weighted_similarity_scores /= sum_abs_weights
    print("Combined weighted similarity scores calculated.")
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
    # Sort by 'Weighted Similarity Score' in descending order
    result_df = result_df.sort_values(by='Weighted Similarity Score', ascending=False).reset_index(drop=True)
    
    print(f"Successfully compiled list of {len(result_df)} books including user-rated books.")
    return result_df

def find_all_books_with_scores(similarity_matrix_npy_filepath, book_identifiers_txt_filepath, user_books, user_ratings):
    """
    Finds similarity scores for all books based on a user's list of rated books,
    including the books the user has rated themselves. The output includes a column
    indicating if the book was rated by the user and their original rating.

    Args:
        similarity_matrix_npy_filepath (str): Path to the NumPy .npy file containing the similarity matrix.
        book_identifiers_txt_filepath (str): Path to the text file containing the book identifiers
                                             (one per line), corresponding to the NPY matrix order.
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

    # 1. Load Similarity Data
    sim_matrix, all_book_identifiers = _load_similarity_data(
        similarity_matrix_npy_filepath, book_identifiers_txt_filepath
    )
    if sim_matrix is None:
        return pd.DataFrame()

    # 2. Normalize Ratings (used for calculating combined scores)
    normalized_user_ratings_np = _normalize_ratings(user_ratings)

    # 3. Map User Books to Matrix Indices (and get original ratings for output)
    user_book_indices_np, _ = _map_user_books_to_indices(
        user_books, user_ratings, all_book_identifiers
    )
    if user_book_indices_np is None: # Means no user books found in the matrix
        return pd.DataFrame()

    # 4. Calculate Combined Weighted Similarity Scores
    combined_weighted_scores = _calculate_combined_weighted_scores(
        sim_matrix, user_book_indices_np, normalized_user_ratings_np, len(all_book_identifiers)
    )
    if combined_weighted_scores is None: # Means sum_abs_weights was 0
        return pd.DataFrame()

    # 5. Prepare the full list including rated books
    full_recommendation_df = _prepare_full_recommendation_list(
        combined_weighted_scores, all_book_identifiers, user_books, user_ratings
    )
    
    return full_recommendation_df


