import pandas as pd # type: ignore
import numpy as np # type: ignore

# --- Define file paths ---
base_output_filename='book_similarity_matrix'
g= 1
similarity_matrix_npy_filepath  = f"{base_output_filename}_{str(g).replace('.', '-')}.npy"
book_identifiers_txt_filepath = f"{base_output_filename}_books_{str(g).replace('.', '-')}.txt"

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

def _normalize_ratings(my_ratings):
    """
    Normalizes user ratings to a 0-1 scale based on the user's own min/max rating.
    If all ratings are identical, they are treated as a positive weight of 1.0.
    """
    user_ratings_np = np.array(my_ratings)
    print(f"User's raw ratings: {my_ratings}")

    normalized_ratings = []

    # Check if all user ratings are identical
    if np.all(user_ratings_np == user_ratings_np[0]):
        # If all original ratings are identical (e.g., all 3s, all 5s, all 1s),
        # treat them as a positive weight of 1.0 for recommendations.
        normalized_ratings = [1.0] * len(my_ratings)
        print(f"All your ratings are identical, treating them as a positive weight of 1 for recommendations.")
    else:
        user_min_rating = user_ratings_np.min()
        user_max_rating = user_ratings_np.max()
        
        # Calculate the effective range of the user's ratings
        user_rating_range = user_max_rating - user_min_rating

        # This check is crucial to prevent division by zero if, despite the `np.all` check above,
        # there's a logical path where `user_rating_range` could be zero (e.g., only one book rated).
        # However, if `np.all` passes, user_rating_range will be > 0.
        if user_rating_range == 0: 
            # This case should ideally be caught by the `np.all` condition
            normalized_ratings = [1] * len(my_ratings) # Fallback to neutral if range is unexpectedly zero
            print(f"Warning: User's rating range is zero despite having varying ratings, normalized to 1. This should have been caught by the 'all identical' check.")
        else:
            # Min-Max scaling using user's own min and max
            # Formula: (x - user_min) / (user_max - user_min)
            for r in my_ratings:
                normalized_ratings.append((r - user_min_rating) / user_rating_range)
    
    normalized_ratings_np = np.array(normalized_ratings)
    print(f"Normalized (0-1 scaled based on user's range) ratings: {normalized_ratings_np}")
    return normalized_ratings_np

def _map_user_books_to_indices(my_books, normalized_ratings_np, all_book_identifiers):
    """
    Maps user's rated books to their corresponding indices in the similarity matrix
    and associates them with their normalized ratings.
    """
    user_book_indices = []
    weighted_user_ratings = []
    
    for i, book_id in enumerate(my_books):
        try:
            idx_in_matrix = all_book_identifiers.index(book_id)
            user_book_indices.append(idx_in_matrix)
            weighted_user_ratings.append(normalized_ratings_np[i])
        except ValueError:
            print(f"Warning: Your rated book '{book_id}' not found in the overall book list. Skipping.")

    if not user_book_indices:
        print("No rated books from your list were found in the similarity matrix. Cannot generate recommendations.")
        return None, None
    
    return np.array(user_book_indices), np.array(weighted_user_ratings)

def _calculate_combined_weighted_scores(sim_matrix, user_book_indices_np, weighted_user_ratings_np, total_books_count):
    """
    Calculates the combined weighted similarity score for all books in the matrix
    based on the user's rated books and their weights.
    """
    combined_weighted_similarity_scores = np.zeros(total_books_count)
    sum_abs_weights = np.sum(np.abs(weighted_user_ratings_np))

    if sum_abs_weights == 0:
        print("All your found rated books resulted in zero-sum normalized ratings. Cannot generate meaningful weighted recommendations.")
        return None

    for i, user_idx in enumerate(user_book_indices_np):
        rating_weight = weighted_user_ratings_np[i]
        similarities_from_this_book = sim_matrix[user_idx, :]
        combined_weighted_similarity_scores += similarities_from_this_book * rating_weight

    combined_weighted_similarity_scores /= sum_abs_weights
    print("Combined weighted similarity scores calculated.")
    return combined_weighted_similarity_scores

def _filter_and_rank_recommendations(combined_weighted_similarity_scores, all_book_identifiers, user_book_indices_np, n):
    """
    Filters out books the user has already read and ranks the remaining books
    by their weighted similarity scores.
    """
    recommendation_mask = np.ones(len(all_book_identifiers), dtype=bool)
    for idx in user_book_indices_np:
        recommendation_mask[idx] = False

    recommendable_books = np.array(all_book_identifiers)[recommendation_mask]
    recommendable_scores = combined_weighted_similarity_scores[recommendation_mask]

    if len(recommendable_books) == 0:
        print("No unread books left to recommend after filtering out your rated books.")
        return pd.DataFrame()

    sorted_indices = np.argsort(recommendable_scores)[::-1]
    top_n_indices = sorted_indices[:min(n, len(recommendable_books))]

    top_n_books = recommendable_books[top_n_indices]
    top_n_scores = recommendable_scores[top_n_indices]

    result_df = pd.DataFrame({
        'Similar Book': top_n_books,
        'Weighted Similarity Score': top_n_scores
    })
    print(f"Successfully found {len(result_df)} similar books.")
    return result_df


def find_n_most_similar_books(similarity_matrix_npy_filepath, book_identifiers_txt_filepath, my_books, my_ratings, n=5):
    """
    Finds the n most similar books based on a user's list of rated books,
    using a weighted average of similarities where ratings are centered around
    the user's mean rating. If all ratings are identical and neutral (e.g., all 3s),
    they are treated as a positive weight of 1.0.

    Args:
        similarity_matrix_npy_filepath (str): Path to the NumPy .npy file containing the similarity matrix.
        book_identifiers_txt_filepath (str): Path to the text file containing the book identifiers
                                             (one per line), corresponding to the NPY matrix order.
        my_books (list of str): A list of book identifiers (e.g., "Title by Author Name") that the user has rated.
        my_ratings (list of int): A list of corresponding ratings for `my_books` (e.g., 1-5 scale).
        n (int): The number of most similar books to return.

    Returns:
        pd.DataFrame: A DataFrame with 'Similar Book' and 'Weighted Similarity Score' columns,
                      sorted by weighted similarity in descending order. Returns an empty DataFrame
                      if no valid rated books are found or an error occurs during loading.
    """
    print(f"\n--- Finding {n} Most Similar Books for Your Rated Library ---")

    if not my_books or not my_ratings or len(my_books) != len(my_ratings):
        print("Error: 'my_books' and 'my_ratings' must be non-empty lists of the same length.")
        return pd.DataFrame()

    # 1. Load Similarity Data
    sim_matrix, all_book_identifiers = _load_similarity_data(
        similarity_matrix_npy_filepath, book_identifiers_txt_filepath
    )
    if sim_matrix is None:
        return pd.DataFrame()

    # 2. Normalize Ratings
    normalized_ratings_np = _normalize_ratings(my_ratings)

    # 3. Map User Books to Matrix Indices
    user_book_indices_np, weighted_user_ratings_np = _map_user_books_to_indices(
        my_books, normalized_ratings_np, all_book_identifiers
    )
    if user_book_indices_np is None: # Means no user books found in the matrix
        return pd.DataFrame()

    # 4. Calculate Combined Weighted Similarity Scores
    combined_weighted_similarity_scores = _calculate_combined_weighted_scores(
        sim_matrix, user_book_indices_np, weighted_user_ratings_np, len(all_book_identifiers)
    )
    if combined_weighted_similarity_scores is None: # Means sum_abs_weights was 0
        return pd.DataFrame()

    # 5. Filter and Rank Recommendations
    recommended_books_df = _filter_and_rank_recommendations(
        combined_weighted_similarity_scores, all_book_identifiers, user_book_indices_np, n
    )
    
    return recommended_books_df


# Example:
sample_my_books = [
        "gideon the ninth, tamsyn muir",
        "the fifth season, n.k. jemisin",
        "the king of attolia, megan whalen turner",
        "iron widow, xiran jay zhao",
        "all systems red, martha wells",
        "witch king, martha wells",
        "deathless, catherynne m. valente",
        "the adventures of amina al-sirafi, shannon chakraborty",
        "the city of brass, s.a. chakraborty",
        "flight of magpies, k.j. charles",
        "ninefox gambit, yoon ha lee",
        "she who became the sun, shelley parker-chan",
        "boyfriend material, alexis hall",
        "the traitor baru cormorant, seth dickinson",
        "a memory called empire, arkady martine",
        "this is how you lose the time war, amal el-mohtar",
        "summer sons, lee mandelo"

    ]

sample_my_ratings = [5, 5, 5, 3, 4, 5, 2, 4, 5, 4, 5, 4, 4, 5, 4, 5, 4]


n_similar = 30

similar_books = find_n_most_similar_books(similarity_matrix_npy_filepath, book_identifiers_txt_filepath, sample_my_books, sample_my_ratings, n_similar)
print(similar_books)
print(similar_books["Similar Book"])