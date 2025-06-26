import pandas as pd
import numpy as np
import os
import io
import random # Import random for shuffling
from tqdm import tqdm # Import tqdm for progress bars
import sys # Import sys for stdout/stderr redirection
from contextlib import redirect_stdout, redirect_stderr 

# --- Import functions from your content-based and collaborative filtering modules ---
try:
    from cb_get_recs_new import find_all_books_with_scores, BASE_FILE_NAME
    from cf_get_recs import load_pretrained_data, generate_user_predictions, \
                            PRETRAINED_DATA_PATH
except ImportError as e:
    tqdm.write(f"Error importing modules: {e}")
    tqdm.write("Please ensure 'cb_get_recs.py' and 'cf_get_recs.py' are in the same directory or correctly configured in your Python path.")
    sys.exit(1) 

# --- Configuration Constants ---
# These constants are primarily for defining default behavior or for cases where not passed.
DEFAULT_NUM_FEATURES = 400
DEFAULT_NEW_USER_REGULARIZATION_STRENGTH = 15
DEFAULT_GENRE_WEIGHT = 1

def _load_recommendation_data(
    genre_weight: int,
    num_features: int,
    new_user_regularization_strength: int
) -> tuple:
    """
    Helper function to load all necessary data and models for both content-based
    and collaborative filtering recommendations.

    Args:
        genre_weight (int): The weight applied to genre similarity for CB model file paths.
        num_features (int): Number of features for the collaborative filtering model.
        new_user_regularization_strength (int): Regularization strength for the new user CF model.

    Returns:
        tuple: Contains all loaded components in the following order:
               (cb_similarity_matrix_npy_filepath, cb_book_identifiers_txt_filepath,
                book_features_loaded, item_bias_loaded, all_book_identifiers_cf,
                mean_book_ratings_loaded, total_books_loaded)
               Returns (None, ..., None) if any critical data loading fails.
    """
    tqdm.write("\n--- Loading All Necessary Recommendation Data ---")

    # --- Content-Based File Paths ---
    # Ensure BASE_FILE_NAME is defined in cb_get_recs.py and accessible
    cb_similarity_matrix_npy_filepath = f"{BASE_FILE_NAME}_{str(genre_weight).replace('.', '-')}.npy"
    cb_book_identifiers_txt_filepath = f"{BASE_FILE_NAME}_books_{str(genre_weight).replace('.', '-')}.txt"

    # Verify if CB files exist before proceeding, though find_all_books_with_scores handles it.
    if not os.path.exists(cb_similarity_matrix_npy_filepath):
        tqdm.write(f"Content-based similarity matrix file not found: {cb_similarity_matrix_npy_filepath}")
        return (None, None, None, None, None, None, None)
    if not os.path.exists(cb_book_identifiers_txt_filepath):
        tqdm.write(f"Content-based book identifiers file not found: {cb_book_identifiers_txt_filepath}")
        return (None, None, None, None, None, None, None)

    # --- Load Collaborative Filtering Pre-trained Data ---
    book_features_loaded, item_bias_loaded, all_book_identifiers_cf, \
    mean_book_ratings_loaded, total_books_loaded = load_pretrained_data(
        num_features, new_user_regularization_strength, PRETRAINED_DATA_PATH
    )

    if (book_features_loaded is None or item_bias_loaded is None or
        all_book_identifiers_cf is None or mean_book_ratings_loaded is None or
        total_books_loaded is None):
        tqdm.write("Failed to load all necessary pre-trained data for collaborative filtering within helper.")
        return (None, None, None, None, None, None, None)

    tqdm.write("--- All Data Loaded Successfully ---")
    return (cb_similarity_matrix_npy_filepath, cb_book_identifiers_txt_filepath,
            book_features_loaded, item_bias_loaded, all_book_identifiers_cf,
            mean_book_ratings_loaded, total_books_loaded)


def _get_unrated_books(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to filter out books that the user has already rated.

    Args:
        combined_df (pd.DataFrame): The DataFrame containing combined CB and CF scores.

    Returns:
        pd.DataFrame: A DataFrame containing only books not rated by the user.
                      Returns an empty DataFrame if no unrated books are found.
    """
    # Ensure 'Rated by User' column exists and is boolean type before filtering
    if 'Rated by User' not in combined_df.columns:
        # If the column is missing, treat all as unrated for safety, but log a warning
        tqdm.write("Warning: 'Rated by User' column missing in combined_df. Assuming all books are unrated.")
        return combined_df.copy() # Return a copy to avoid modifying the original DataFrame

    # Convert to boolean and fill NaNs (if any)
    combined_df['Rated by User'] = combined_df['Rated by User'].astype('boolean').fillna(False)
    unrated_books = combined_df[~combined_df['Rated by User']].copy()
    if unrated_books.empty:
        tqdm.write("No unrated books available for recommendation.")
    return unrated_books


def _process_user_ratings_file(file_path: str) -> pd.DataFrame:
    """
    Helper function to parse user ratings from a CSV file, combine title and author,
    and organize data into a DataFrame where each row is a user
    and contains lists of their book identifications and ratings.

    Args:
        file_path (str): The path to the CSV file containing user ratings.

    Returns:
        pd.DataFrame: A DataFrame where each row represents a unique user and contains:
                      - 'User_id': The ID of the user.
                      - 'Book_Identifications_List': A list of book identifiers rated by the user.
                      - 'Ratings_List': A list of corresponding ratings by the user.
                      Returns an empty DataFrame if file reading or processing fails.
              Example:
                     User_id           Book_Identifications_List            Ratings_List
              0      1494253      [the woman in the window, a.j. finn]      [3.0]
              1      126589   [shopaholic takes manhattan, sophie kinsella] [Invalid rating handled]
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        tqdm.write(f"Error: CSV file not found at {file_path}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        tqdm.write(f"Error: CSV file at {file_path} is empty.")
        return pd.DataFrame()
    except Exception as e:
        tqdm.write(f"An error occurred while reading the CSV file {file_path}: {e}")
        return pd.DataFrame()

    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()

    # Ensure 'Title', 'Author', 'Rating', 'User_id' columns exist
    required_cols = ['Title', 'Author', 'Rating', 'User_id']
    if not all(col in df.columns for col in required_cols):
        tqdm.write(f"Error: Missing one or more required columns ({required_cols}) in the CSV file.")
        return pd.DataFrame()

    # 1. Combines Title and Author into a string of the form "{Title}, {Author}"
    # and calls this column Book Identification
    df['Book Identification'] = df.apply(
        lambda row: f"{row['Title'].strip()}, {row['Author'].strip()}"
        if pd.notna(row['Title']) and pd.notna(row['Author'])
        else np.nan, axis=1
    )

    # Clean 'Rating' column: replace 'Invalid rating', with NaN and convert to numeric
    df['Rating'] = pd.to_numeric(
        df['Rating'].replace('Invalid rating', np.nan), errors='coerce'
    )

    # Drop rows where 'Book Identification' or 'Rating' is NaN (e.g., from invalid entries)
    df.dropna(subset=['Book Identification', 'Rating'], inplace=True)

    # Group by 'User_id' and aggregate book identifications and ratings into lists
    user_data_df = df.groupby('User_id').agg(
        Book_Identifications_List=('Book Identification', list),
        Ratings_List=('Rating', list)
    ).reset_index()

    # Convert User_id to string for consistency
    user_data_df['User_id'] = user_data_df['User_id'].astype(str)

    return user_data_df


def _split_user_data_for_testing(user_data_df: pd.DataFrame, validation_split_ratio: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the user data into training and validation DataFrames,
    maintaining the structure where each row is a user with lists of books and ratings.
    Each user's rated books are randomly split into two disjoint halves.

    Args:
        user_data_df (pd.DataFrame): A DataFrame where each row is a user and contains
                                     'User_id', 'Book_Identifications_List', and 'Ratings_List'.
        validation_split_ratio (float): The fraction of data to reserve for the validation set (0.0 to 1.0).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
                                           (training_df, validation_df).
                                           Each DataFrame has 'User_id', 'Book_Identifications_List',
                                           and 'Ratings_List' columns.
    """
    training_data_rows = []
    validation_data_rows = []

    if user_data_df.empty:
        tqdm.write("Input user_data_df is empty, returning empty training and validation DataFrames.")
        return pd.DataFrame(columns=['User_id', 'Book_Identifications_List', 'Ratings_List']), \
               pd.DataFrame(columns=['User_id', 'Book_Identifications_List', 'Ratings_List'])

    tqdm.write("\n--- Splitting User Data for Testing ---")
    for index, row in tqdm(user_data_df.iterrows(), total=len(user_data_df), desc="Splitting User Data", unit="user"):
        user_id = row['User_id']
        book_ids = row['Book_Identifications_List']
        ratings = row['Ratings_List']

        # Pair books and ratings and shuffle them
        paired_data = list(zip(book_ids, ratings))
        random.shuffle(paired_data)

        # Determine split point
        num_items = len(paired_data)
        # Ensure at least one item for training and validation if possible
        if num_items < 2:
            # tqdm.write(f"User {user_id} has fewer than 2 items, skipping split.") # Suppress for cleaner bar
            continue # Skip users with too few items to split

        split_point = max(1, int(num_items * (1 - validation_split_ratio))) # Ensure at least 1 for training
        
        # Split into training and validation
        training_set = paired_data[:split_point]
        validation_set = paired_data[split_point:]

        # Append to respective lists for the current user's row
        training_data_rows.append({
            'User_id': user_id,
            'Book_Identifications_List': [item[0] for item in training_set],
            'Ratings_List': [item[1] for item in training_set]
        })
        validation_data_rows.append({
            'User_id': user_id,
            'Book_Identifications_List': [item[0] for item in validation_set],
            'Ratings_List': [item[1] for item in validation_set]
        })

    training_df = pd.DataFrame(training_data_rows)
    validation_df = pd.DataFrame(validation_data_rows)

    expected_cols = ['User_id', 'Book_Identifications_List', 'Ratings_List']
    
    if training_df.empty:
        training_df = pd.DataFrame(columns=expected_cols)
    else:
        training_df = training_df[expected_cols]

    if validation_df.empty:
        validation_df = pd.DataFrame(columns=expected_cols)
    else:
        validation_df = validation_df[expected_cols]


    tqdm.write(f"\n--- User Data Split for Testing Complete ---")
    tqdm.write(f"Total users in original data: {len(user_data_df)}")
    tqdm.write(f"Users with enough data for split (Training): {len(training_df)}")
    tqdm.write(f"Users with enough data for split (Validation): {len(validation_df)}")
    total_original_ratings = user_data_df['Book_Identifications_List'].apply(len).sum()
    total_split_ratings = training_df['Book_Identifications_List'].apply(len).sum() + \
                          validation_df['Book_Identifications_List'].apply(len).sum()
    tqdm.write(f"Total original ratings: {total_original_ratings}")
    tqdm.write(f"Total ratings in split data: {total_split_ratings}")


    return training_df, validation_df


def _get_relevant_books_from_validation(validation_data_df: pd.DataFrame, relevance_threshold: float = 3.0) -> pd.DataFrame:
    """
    Extracts a list of "relevant" books for each user from the validation DataFrame.
    A book is considered relevant if its rating is greater than or equal to the relevance_threshold.

    Args:
        validation_data_df (pd.DataFrame): A DataFrame where each row is a user and contains
                                           'User_id', 'Book_Identifications_List', and 'Ratings_List' (validation data).
        relevance_threshold (float): The minimum rating for a book to be considered "relevant".

    Returns:
        pd.DataFrame: A DataFrame with 'User_id' and 'Relevant_Books_List' columns.
                      'Relevant_Books_List' contains a list of book identifiers that the user rated >= relevance_threshold.
    """
    relevant_books_data = []

    if validation_data_df.empty:
        tqdm.write("Validation data DataFrame is empty. Cannot extract relevant books.")
        return pd.DataFrame(columns=['User_id', 'Relevant_Books_List'])

    tqdm.write(f"\n--- Extracting Relevant Books from Validation Data (threshold >= {relevance_threshold}) ---")

    for index, row in tqdm(validation_data_df.iterrows(), total=len(validation_data_df), desc="Extracting Relevant Books", unit="user"):
        user_id = row['User_id']
        book_ids = row['Book_Identifications_List']
        ratings = row['Ratings_List']

        relevant_books_for_user = []
        for book_id, rating in zip(book_ids, ratings):
            if rating >= relevance_threshold:
                relevant_books_for_user.append(book_id)
        
        relevant_books_data.append({
            'User_id': user_id,
            'Relevant_Books_List': relevant_books_for_user
        })
    
    relevant_books_df = pd.DataFrame(relevant_books_data)
    tqdm.write("--- Relevant Books Extraction Complete ---")
    return relevant_books_df


def get_combined_recommendations(
    user_books: list[str],
    user_ratings: list[float],
    loaded_data_tuple: tuple,
    new_user_regularization_strength: int = DEFAULT_NEW_USER_REGULARIZATION_STRENGTH
) -> pd.DataFrame | None:
    """
    Generates combined book recommendations using both content-based similarity
    and collaborative filtering prediction models.

    Args:
        user_books (list): A list of book identifiers that the user has rated.
        user_ratings (list): A list of corresponding ratings for `user_books`.
        loaded_data_tuple (tuple): A tuple containing all pre-loaded data components
                                   from _load_recommendation_data.
        new_user_regularization_strength (int): Regularization strength for the new user CF model.

    Returns:
        pd.DataFrame: Contains 'Book Identifier', 'CB_Weighted_Similarity_Score',
                      'CF_Predicted_Rating', 'Rated by User', and 'Original Rating' columns.
                      Returns None if data loading or model prediction fails for either source.
    """
    # Use StringIO as a temporary buffer to capture output
    temp_stdout = io.StringIO()
    temp_stderr = io.StringIO()

    with redirect_stdout(temp_stdout), redirect_stderr(temp_stderr):
        tqdm.write(f"--- Starting Combined Recommendation Generation for Settings ---")
        tqdm.write(f"  User Reg Strength: {new_user_regularization_strength}")

        # --- Unpack all necessary data from the provided tuple ---
        (cb_similarity_matrix_npy_filepath, cb_book_identifiers_txt_filepath,
         book_features_loaded, item_bias_loaded, all_book_identifiers_cf,
         mean_book_ratings_loaded, total_books_loaded) = loaded_data_tuple

        if (cb_similarity_matrix_npy_filepath is None or
            book_features_loaded is None): # Check for any None indicating failure from loaded tuple
            tqdm.write("Provided loaded_data_tuple indicates failure or incomplete data. Exiting.")
            return None 

        # --- 1. Get Content-Based Recommendations ---
        tqdm.write("--- Generating Content-Based Recommendations ---")
        cb_df = find_all_books_with_scores(
            cb_similarity_matrix_npy_filepath,
            cb_book_identifiers_txt_filepath,
            user_books,
            user_ratings
        )

        if cb_df is None or cb_df.empty:
            tqdm.write("Content-based recommendation generation failed or returned an empty DataFrame.")
            cb_df = pd.DataFrame(columns=['Book Identifier', 'Weighted Similarity Score'])
        
        # Rename columns for clarity in the combined DataFrame
        cb_df = cb_df.rename(columns={
            'Weighted Similarity Score': 'CB_Weighted_Similarity_Score'
        })
        cb_df = cb_df[['Book Identifier', 'CB_Weighted_Similarity_Score']]

        # --- 2. Get Collaborative Filtering Predictions ---
        tqdm.write("--- Generating Collaborative Filtering Predictions ---")
        cf_df = generate_user_predictions(
            book_features_loaded,
            item_bias_loaded,
            mean_book_ratings_loaded,
            all_book_identifiers_cf,
            total_books_loaded, 
            user_books,
            user_ratings,
            new_user_regularization_strength
        )

        if cf_df is None or cf_df.empty:
            tqdm.write("Collaborative filtering prediction generation failed or returned an empty DataFrame.")
            cf_df = pd.DataFrame(columns=['Book Identifier', 'Predicted Rating', 'Rated by User', 'Original Rating'])


        # Rename columns for clarity in the combined DataFrame
        cf_df = cf_df.rename(columns={
            'Predicted Rating': 'CF_Predicted_Rating'
        })
        
        # --- 3. Combine the DataFrames ---
        tqdm.write("--- Combining Recommendations ---")
        # Perform an outer merge to ensure all books from both CB and CF are included
        combined_df = pd.merge(
            cb_df,
            cf_df,
            on='Book Identifier',
            how='outer',
            suffixes=('_CB', '_CF') 
        )

        # --- Fill NaN scores with 0 and ensure 'Rated by User' and 'Original Rating' are present ---
        # Initialize 'Rated by User' and 'Original Rating' if they are missing after merge
        if 'Rated by User' not in combined_df.columns:
            combined_df['Rated by User'] = False # Default to False if column is missing
        if 'Original Rating' not in combined_df.columns:
            combined_df['Original Rating'] = np.nan # Default to NaN if column is missing

        # Convert 'Rated by User' to boolean and handle NaNs from CF predictions if any
        combined_df['Rated by User'] = combined_df['Rated by User'].astype('boolean').fillna(False)

        # Fill NaN scores with 0 (for books only in one source or not predicted by CF)
        combined_df['CB_Weighted_Similarity_Score'] = combined_df['CB_Weighted_Similarity_Score'].fillna(0)
        combined_df['CF_Predicted_Rating'] = combined_df['CF_Predicted_Rating'].fillna(0)


        expected_cols_combined = [
            'Book Identifier', 'CB_Weighted_Similarity_Score', 'CF_Predicted_Rating',
            'Rated by User', 'Original Rating'
        ]
        for col in expected_cols_combined:
            if col not in combined_df.columns:
                combined_df[col] = np.nan 

        combined_df = combined_df[expected_cols_combined] 

        combined_df = combined_df.sort_values(
            by=['CF_Predicted_Rating', 'CB_Weighted_Similarity_Score'],
            ascending=[False, False]
        ).reset_index(drop=True)

        tqdm.write("--- Combined Recommendations Generated Successfully ---")
        return combined_df



def recommend_fraction_of_top_n(
    combined_df: pd.DataFrame,
    output_limit: int = 30,
    cb_fraction: float = 0.5
) -> list[str]:
    """
    Recommends books by taking a specified fraction of the top N from content-based
    and the remaining fraction from collaborative filtering, then combining them.
    Prioritizes books that appear in BOTH lists.
    Only considers books not rated by the user.

    Args:
        combined_df (pd.DataFrame): The DataFrame containing combined CB and CF scores.
        output_limit (int): The total number of top recommendations to consider from each method.
        cb_fraction (float): The fraction (0.0 to 1.0) of 'output_limit' to take from the content-based method.
                             The remaining fraction (1 - cb_fraction) will be taken from CF.

    Returns:
        list[str]: A list of recommended book identifiers, sorted with duplicates prioritized.
    """
    unrated_books = _get_unrated_books(combined_df) 
    if unrated_books.empty:
        return []

    num_to_take_cb = int(output_limit * cb_fraction)
    num_to_take_cf = int(output_limit * (1 - cb_fraction)) 

    # Get top from Content-Based
    top_cb_books = unrated_books.sort_values(
        by='CB_Weighted_Similarity_Score', ascending=False
    ).head(num_to_take_cb)

    # Get top from Collaborative Filtering
    top_cf_books = unrated_books.sort_values(
        by='CF_Predicted_Rating', ascending=False
    ).head(num_to_take_cf)

    # Identify books that are in both top lists
    cb_ids = set(top_cb_books['Book Identifier'])
    cf_ids = set(top_cf_books['Book Identifier'])
    
    # Books recommended by both methods
    books_recommended_by_both = list(cb_ids.intersection(cf_ids))

    # Add a flag for prioritization
    unrated_books['Recommended_By_Both'] = unrated_books['Book Identifier'].isin(books_recommended_by_both)

    # Filter unrated_books to only include books from either of the top N lists,
    # or books from the intersection.
    candidate_books = unrated_books[
        unrated_books['Book Identifier'].isin(cb_ids.union(cf_ids))
    ].copy()

    # Sort: First by 'Recommended_By_Both' (True first) then by 'CB_Weighted_Similarity_Score'
    combined_top_n_prioritized = candidate_books.sort_values(
        by=['Recommended_By_Both', 'CB_Weighted_Similarity_Score'],
        ascending=[False, False]
    ).reset_index(drop=True)

    # Return the 'Book Identifier' column as a list, limited to output_limit
    return combined_top_n_prioritized['Book Identifier'].head(output_limit).tolist()


def recommend_cb_filtered_by_cf(
    combined_df: pd.DataFrame,
    cb_initial_top_n: int = 200,
    output_limit: int = 30
) -> list[str]:
    """
    Recommends books by first taking the top N from content-based recommendations,
    then filtering and re-ranking that list based on collaborative filtering scores.
    Only considers books not rated by the user.

    Args:
        combined_df (pd.DataFrame): The DataFrame containing combined CB and CF scores.
        cb_initial_top_n (int): The number of top content-based recommendations to consider.
        output_limit (int): The maximum number of book identifiers to return in the list.

    Returns:
        list[str]: A list of recommended book identifiers, sorted by CF_Predicted_Rating
                   from the initially CB-filtered list.
    """
    unrated_books = _get_unrated_books(combined_df) 
    if unrated_books.empty:
        return []

    top_cb_initial = unrated_books.sort_values(
        by='CB_Weighted_Similarity_Score', ascending=False
    ).head(cb_initial_top_n)

    cf_filtered_cb = top_cb_initial.sort_values(
        by='CF_Predicted_Rating', ascending=False
    ).reset_index(drop=True)

    # Return the 'Book Identifier' column as a list, limited to output_limit
    return cf_filtered_cb['Book Identifier'].head(output_limit).tolist()


def recommend_by_multiplying_scores(
    combined_df: pd.DataFrame,
    output_limit: int = 30
) -> list[str]:
    """
    Recommends books by creating a new combined score by multiplying
    the content-based similarity score and the collaborative filtering predicted rating.
    Only considers books not rated by the user.

    Args:
        combined_df (pd.DataFrame): The DataFrame containing combined CB and CF scores.
        output_limit (int): The maximum number of book identifiers to return in the list.

    Returns:
        list[str]: A list of recommended book identifiers, sorted by the new multiplied score.
    """
    unrated_books = _get_unrated_books(combined_df) 
    if unrated_books.empty:
        return []

    # Create the combined score by multiplying the two scores
    unrated_books['Combined_Multiplied_Score'] = (
        unrated_books['CB_Weighted_Similarity_Score'] * unrated_books['CF_Predicted_Rating']
    )

    # Sort by the new combined score
    multiplied_recommendations = unrated_books.sort_values(
        by='Combined_Multiplied_Score', ascending=False
    ).reset_index(drop=True)

    # Return the 'Book Identifier' column as a list, limited to output_limit
    return multiplied_recommendations['Book Identifier'].head(output_limit).tolist()


def recommend_hybrid_strategy_2_and_1(
    combined_df: pd.DataFrame,
    n: int = 100, 
    cb_fraction: float = 0.5,
    cb_initial_top_n_for_filtered_list: int = 200,
    output_limit: int = 30
) -> list[str]:
    """
    Recommends books by combining:
    1. A fraction of pure top N content-based recommendations.
    2. The remaining fraction from a list of top content-based books re-ranked by collaborative filtering.
    Prioritizes books appearing in both fractions.
    Only considers books not rated by the user.

    Args:
        combined_df (pd.DataFrame): The DataFrame containing combined CB and CF scores.
        n (int): The total number of top recommendations to consider across both fractions.
        cb_fraction (float): The fraction (0.0 to 1.0) of 'n' to take as pure content-based.
                             The remaining fraction (1 - cb_fraction) will be taken from the filtered list.
        cb_initial_top_n_for_filtered_list (int): The initial top N for the content-based books
                                                  that are then filtered by CF (used for the second fraction).
        output_limit (int): The maximum number of book identifiers to return in the list.

    Returns:
        list[str]: A list of recommended book identifiers.
    """
    unrated_books = _get_unrated_books(combined_df)
    if unrated_books.empty:
        return []

    # Calculate number of books for each fraction
    num_to_take_pure_cb = int(n * cb_fraction)
    num_to_take_filtered_list = int(n * (1 - cb_fraction))

    # --- 1. Get Pure Content-Based Top N ---
    pure_cb_recs = unrated_books.sort_values(
        by='CB_Weighted_Similarity_Score', ascending=False
    ).head(num_to_take_pure_cb)

    # --- 2. Get Filtered List (Strategy 2's output) ---

    top_cb_initial_for_filter = unrated_books.sort_values(
        by='CB_Weighted_Similarity_Score', ascending=False
    ).head(cb_initial_top_n_for_filtered_list)

    filtered_by_cf_recs = top_cb_initial_for_filter.sort_values(
        by='CF_Predicted_Rating', ascending=False
    ).head(num_to_take_filtered_list)


    # --- Combine and Prioritize ---
    # Combine the two sets of recommendations
    combined_candidates = pd.concat([pure_cb_recs, filtered_by_cf_recs]).drop_duplicates(
        subset=['Book Identifier']
    ).copy() # Use .copy() to avoid SettingWithCopyWarning

    if combined_candidates.empty:
        return []

    # Identify books that appeared in both sub-lists (pure CB and filtered list)
    pure_cb_ids = set(pure_cb_recs['Book Identifier'])
    filtered_list_ids = set(filtered_by_cf_recs['Book Identifier'])
    books_recommended_by_both = list(pure_cb_ids.intersection(filtered_list_ids))

    combined_candidates['Recommended_By_Both'] = combined_candidates['Book Identifier'].isin(books_recommended_by_both)

    # Sort: First prioritize books recommended by both, then by CF_Predicted_Rating, then by CB_Weighted_Similarity_Score
    hybrid_recommendations = combined_candidates.sort_values(
        by=['Recommended_By_Both', 'CF_Predicted_Rating', 'CB_Weighted_Similarity_Score'],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return hybrid_recommendations['Book Identifier'].head(output_limit).tolist()


def recommend_random_books(
    combined_df: pd.DataFrame,
    output_limit: int = 30
) -> list[str]:
    """
    Generates a list of random book recommendations from the unrated books.

    Args:
        combined_df (pd.DataFrame): The DataFrame containing all books, including those rated by the user.
        output_limit (int): The maximum number of random book identifiers to return.

    Returns:
        list[str]: A list of randomly selected unrated book identifiers.
    """
    unrated_books = _get_unrated_books(combined_df)
    if unrated_books.empty:
        tqdm.write("No unrated books available for random recommendation.")
        return []
    
    # Shuffle the unrated books and take the top 'output_limit'
    # Use .sample(frac=1) for a true shuffle and then .head(output_limit)
    # Ensure 'Book Identifier' column exists before sampling
    if 'Book Identifier' not in unrated_books.columns:
        tqdm.write("Error: 'Book Identifier' column missing in unrated_books DataFrame for random recommendations.")
        return []

    random_recs_df = unrated_books.sample(frac=1, random_state=42).head(output_limit) # Use random_state for reproducibility
    return random_recs_df['Book Identifier'].tolist()


def generate_all_recommendations_for_users(
    training_data_df: pd.DataFrame,
    loaded_data_tuple: tuple,
    new_user_regularization_strength: int = DEFAULT_NEW_USER_REGULARIZATION_STRENGTH,
    output_limit_per_strategy: int = 20 # Default output limit for each recommendation strategy
) -> pd.DataFrame:
    """
    Generates book recommendations for each user in the training data using all
    defined recommendation strategies (excluding the hybrid strategy).

    Args:
        training_data_df (pd.DataFrame): A DataFrame where each row is a user
                                         and contains 'User_id', 'Book_Identifications_List',
                                         and 'Ratings_List' (training data).
        loaded_data_tuple (tuple): The pre-loaded data and models from _load_recommendation_data.
        new_user_regularization_strength (int): Regularization strength for the new user CF model.
        output_limit_per_strategy (int): The maximum number of recommendations to generate
                                         for each strategy for each user.

    Returns:
        pd.DataFrame: A DataFrame with 'User_id' and columns for each recommendation strategy,
                      where each column contains a list of recommended book identifiers.
    """
    all_user_recommendations = []

    tqdm.write("\n--- Generating All Recommendations for Users ---")

    if training_data_df.empty:
        tqdm.write("Training data DataFrame is empty. Cannot generate recommendations.")
        return pd.DataFrame(columns=[
            'User_id',
            'Rec_CB_Only',
            'Rec_CF_Only',
            'Rec_CB_Filtered_by_CF',
            'Rec_By_Multiplying_Scores',
            'Rec_Random' 
        ])

    for index, user_row in tqdm(training_data_df.iterrows(), total=len(training_data_df), desc="Generating User Recs", unit="user"):
        user_id = user_row['User_id']
        # Explicitly cast to list to avoid potential ValueError from numpy/pandas array-like behavior
        user_books_training = list(user_row['Book_Identifications_List']) 
        user_ratings_training = list(user_row['Ratings_List'])

        # Get combined scores for the current user's training data
        combined_df = get_combined_recommendations(
            user_books=user_books_training,
            user_ratings=user_ratings_training,
            loaded_data_tuple=loaded_data_tuple,
            new_user_regularization_strength=new_user_regularization_strength
        )

        user_recs = {'User_id': user_id}

        if combined_df is None or combined_df.empty:
            tqdm.write(f"Skipping recommendations for user {user_id} due to empty or failed combined_df.")
            # Ensure all recommendation columns are present, even if empty
            user_recs['Rec_CB_Only'] = []
            user_recs['Rec_CF_Only'] = []
            user_recs['Rec_CB_Filtered_by_CF'] = []
            user_recs['Rec_By_Multiplying_Scores'] = []
            user_recs['Rec_Random'] = [] 
        else:
            user_recs['Rec_CB_Only'] = recommend_fraction_of_top_n(
                combined_df, output_limit=output_limit_per_strategy, cb_fraction=1.0
            )
            user_recs['Rec_CF_Only'] = recommend_fraction_of_top_n(
                combined_df, output_limit=output_limit_per_strategy, cb_fraction=0.0
            )
            user_recs['Rec_CB_Filtered_by_CF'] = recommend_cb_filtered_by_cf(
                combined_df, output_limit=output_limit_per_strategy
            )
            user_recs['Rec_By_Multiplying_Scores'] = recommend_by_multiplying_scores(
                combined_df, output_limit=output_limit_per_strategy
            )
            user_recs['Rec_Random'] = recommend_random_books( 
                combined_df, output_limit=output_limit_per_strategy
            )
        
        all_user_recommendations.append(user_recs)

    recommendations_df = pd.DataFrame(all_user_recommendations)
    tqdm.write("\n--- All User Recommendations Generated Successfully ---")
    return recommendations_df


def evaluate_recommendations_overlap(
    user_recommendations_df: pd.DataFrame,
    relevant_validation_books_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compares generated recommendation lists with relevant books in the validation set.

    Args:
        user_recommendations_df (pd.DataFrame): DataFrame with 'User_id' and columns
                                                for each recommendation strategy (lists of book identifiers).
                                                Example: 'Rec_CB_Only', 'Rec_CF_Only', etc.
        relevant_validation_books_df (pd.DataFrame): DataFrame with 'User_id' and
                                                     'Relevant_Books_List' (list of book identifiers).

    Returns:
        pd.DataFrame: A DataFrame where each row represents a user, and columns indicate
                      the number of overlapping books for each strategy, plus the total
                      number of relevant books in the validation set for that user.
                      Columns: 'User_id', 'Total_Relevant_Books', 'Overlap_Rec_CB_Only',
                      'Overlap_Rec_CF_Only', 'Overlap_Rec_CB_Filtered_by_CF',
                      'Overlap_Rec_By_Multiplying_Scores'.
    """
    evaluation_results = []

    # Merge the two dataframes on User_id
    merged_df = pd.merge(
        user_recommendations_df,
        relevant_validation_books_df,
        on='User_id',
        how='left' # Use left merge to keep all users from recommendations_df
    )

    if merged_df.empty:
        tqdm.write("Merged DataFrame is empty. No recommendations or relevant books to evaluate.")
        return pd.DataFrame(columns=[
            'User_id', 'Total_Relevant_Books', 'Overlap_Rec_CB_Only',
            'Overlap_Rec_CF_Only', 'Overlap_Rec_CB_Filtered_by_CF',
            'Overlap_Rec_By_Multiplying_Scores'
        ])

    tqdm.write("\n--- Evaluating Recommendation Overlap with Relevant Validation Books ---")

    reco_cols = [col for col in user_recommendations_df.columns if col.startswith('Rec_')] 

    for index, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Evaluating Overlap", unit="user"):
        user_id = row['User_id']
        # Ensure Relevant_Books_List is a list (handle potential NaN from left merge)
        relevant_books = set(row['Relevant_Books_List'] if isinstance(row['Relevant_Books_List'], list) else [])
        total_relevant = len(relevant_books)

        user_eval = {'User_id': user_id, 'Total_Relevant_Books': total_relevant}

        for col in reco_cols:
            # Ensure the recommendation list is a list (handle potential NaN)
            recs_list = set(row[col] if isinstance(row[col], list) else [])
            overlap_count = len(recs_list.intersection(relevant_books))
            user_eval[f'Overlap_{col}'] = overlap_count
        
        evaluation_results.append(user_eval)
    
    evaluation_df = pd.DataFrame(evaluation_results)
    tqdm.write("--- Recommendation Overlap Evaluation Complete ---")
    return evaluation_df
