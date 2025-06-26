import pandas as pd
import numpy as np
import os

# --- Import functions from your content-based and collaborative filtering modules ---
try:
    # Use relative imports (the '.' signifies the current package)
    from cb_get_recs import find_all_books_with_scores, BASE_FILE_NAME
    from cf_get_recs import load_pretrained_data, generate_user_predictions, \
                                PRETRAINED_DATA_PATH
except ImportError as e:
    print(f"Error importing modules within combined.py: {e}")
    print("Please ensure 'cb_get_recs.py' and 'cf_get_recs.py' are in the same directory as 'combined.py' and that 'combined_model' is treated as a Python package (contains an __init__.py file).")
    exit()

# --- Configuration Constants ---
# These constants are primarily for defining default behavior or for cases where not passed.
DEFAULT_NUM_FEATURES = 600
DEFAULT_NEW_USER_REGULARIZATION_STRENGTH = 5
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
    print("\n--- Loading All Necessary Recommendation Data ---")

    # --- Content-Based File Paths ---
    cb_similarity_matrix_npy_filepath = f"{BASE_FILE_NAME}_{str(genre_weight).replace('.', '-')}.npy"
    cb_book_identifiers_txt_filepath = f"{BASE_FILE_NAME}_books_{str(genre_weight).replace('.', '-')}.txt"

    # Verify if CB files exist before proceeding, though find_all_books_with_scores handles it.
    if not os.path.exists(cb_similarity_matrix_npy_filepath):
        print(f"Content-based similarity matrix file not found: {cb_similarity_matrix_npy_filepath}")
        return (None, None, None, None, None, None, None)
    if not os.path.exists(cb_book_identifiers_txt_filepath):
        print(f"Content-based book identifiers file not found: {cb_book_identifiers_txt_filepath}")
        return (None, None, None, None, None, None, None)

    # --- Load Collaborative Filtering Pre-trained Data ---
    book_features_loaded, item_bias_loaded, all_book_identifiers_cf, \
    mean_book_ratings_loaded, total_books_loaded = load_pretrained_data(
        num_features, new_user_regularization_strength, PRETRAINED_DATA_PATH
    )

    if (book_features_loaded is None or item_bias_loaded is None or
        all_book_identifiers_cf is None or mean_book_ratings_loaded is None or
        total_books_loaded is None):
        print("Failed to load all necessary pre-trained data for collaborative filtering within helper.")
        return (None, None, None, None, None, None, None)

    print("--- All Data Loaded Successfully ---")
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
    unrated_books = combined_df[~combined_df['Rated by User']].copy()
    if unrated_books.empty:
        print("No unrated books available for recommendation.")
    return unrated_books


def get_combined_recommendations(
    user_books: list[str],
    user_ratings: list[float],
    genre_weight: int = DEFAULT_GENRE_WEIGHT,
    num_features: int = DEFAULT_NUM_FEATURES,
    new_user_regularization_strength: int = DEFAULT_NEW_USER_REGULARIZATION_STRENGTH
) -> pd.DataFrame | None: # Updated return type hint
    """
    Generates combined book recommendations using both content-based similarity
    and collaborative filtering prediction models.

    Args:
        user_books (list): A list of book identifiers that the user has rated.
        user_ratings (list): A list of corresponding ratings for `user_books`.
        genre_weight (int): The weight applied to genre similarity in content-based model.
        num_features (int): Number of features for the collaborative filtering model.
        new_user_regularization_strength (int): Regularization strength for the new user CF model.

    Returns:
        pd.DataFrame: Contains 'Book Identifier', 'CB_Weighted_Similarity_Score',
                      'CF_Predicted_Rating', 'Rated by User', and 'Original Rating' columns.
                      Returns None if data loading or model prediction fails for either source.
    """
    print(f"\n--- Starting Combined Recommendation Generation for Settings ---")
    print(f"  Genre Weight: {genre_weight}, Num Features: {num_features}, User Reg Strength: {new_user_regularization_strength}")

    # --- Load all necessary data using the helper function ---
    (cb_similarity_matrix_npy_filepath, cb_book_identifiers_txt_filepath,
     book_features_loaded, item_bias_loaded, all_book_identifiers_cf,
     mean_book_ratings_loaded, total_books_loaded) = _load_recommendation_data(
        genre_weight, num_features, new_user_regularization_strength
    )

    if (cb_similarity_matrix_npy_filepath is None or
        book_features_loaded is None): # Check for any None indicating failure
        print("Failed to load necessary data for combined recommendations. Exiting.")
        return None

    # --- 1. Get Content-Based Recommendations ---
    print("\n--- Generating Content-Based Recommendations ---")
    cb_df = find_all_books_with_scores(
        cb_similarity_matrix_npy_filepath,
        cb_book_identifiers_txt_filepath,
        user_books,
        user_ratings
    )

    if cb_df is None or cb_df.empty:
        print("Content-based recommendation generation failed or returned an empty DataFrame.")
        return None
    
    # Rename columns for clarity in the combined DataFrame
    cb_df = cb_df.rename(columns={
        'Weighted Similarity Score': 'CB_Weighted_Similarity_Score'
    })
    cb_df = cb_df[['Book Identifier', 'CB_Weighted_Similarity_Score']]

    # --- 2. Get Collaborative Filtering Predictions ---
    print("\n--- Generating Collaborative Filtering Predictions ---")
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
        print("Collaborative filtering prediction generation failed or returned an empty DataFrame.")
        return None

    # Rename columns for clarity in the combined DataFrame
    cf_df = cf_df.rename(columns={
        'Predicted Rating': 'CF_Predicted_Rating'
    })
    
    # --- 3. Combine the DataFrames ---
    print("\n--- Combining Recommendations ---")
    combined_df = pd.merge(
        cb_df,
        cf_df,
        on='Book Identifier',
        how='outer',
        suffixes=('_CB', '_CF')
    )

    # --- Convert 'Rated by User' to boolean and handle NaNs ---
    # This addresses the FutureWarning regarding implicit downcasting by explicitly
    # converting to the nullable boolean dtype first, then filling NaNs.
    combined_df['Rated by User'] = combined_df['Rated by User'].astype('boolean').fillna(False)

    # --- Fill NaN scores with 0 ---
    combined_df['CB_Weighted_Similarity_Score'] = combined_df['CB_Weighted_Similarity_Score'].fillna(0)
    combined_df['CF_Predicted_Rating'] = combined_df['CF_Predicted_Rating'].fillna(0)

    combined_df = combined_df.sort_values(
        by=['CF_Predicted_Rating', 'CB_Weighted_Similarity_Score'],
        ascending=[False, False]
    ).reset_index(drop=True)

    print("\n--- Combined Recommendations Generated Successfully ---")
    return combined_df


def recommend_fraction_of_top_n(
    combined_df: pd.DataFrame,
    n: int = 100,
    cb_fraction: float = 0.5
) -> list[str]:
    """
    Recommends books by taking a specified fraction of the top N from content-based
    and the remaining fraction from collaborative filtering, then combining them.
    Prioritizes books that appear in BOTH lists.
    Only considers books not rated by the user.

    Args:
        combined_df (pd.DataFrame): The DataFrame containing combined CB and CF scores.
        n (int): The total number of top recommendations to consider from each method.
        cb_fraction (float): The fraction (0.0 to 1.0) of 'n' to take from the content-based method.
                             The remaining fraction (1 - cb_fraction) will be taken from CF.

    Returns:
        list[str]: A list of recommended book identifiers, sorted with duplicates prioritized.
    """
    unrated_books = _get_unrated_books(combined_df) # Use helper
    if unrated_books.empty:
        return []

    num_to_take_cb = int(n * cb_fraction)
    num_to_take_cf = int(n * (1 - cb_fraction)) 

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
    return combined_top_n_prioritized['Book Identifier'].head(n).tolist()


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
    unrated_books = _get_unrated_books(combined_df) # Use helper
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
    unrated_books = _get_unrated_books(combined_df) # Use helper
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
    unrated_books = _get_unrated_books(combined_df) # Use helper
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
    # Re-use the logic from recommend_cb_filtered_by_cf but apply it to the unrated_books subset
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


###For display only:

def print_as_numbered_list(input_list):
    """
    Takes a list and prints its elements as a numbered list.

    Args:
        input_list (list): The list to be printed.
    """
    if not isinstance(input_list, list):
        print("Error: Input must be a list.")
        return

    if not input_list:
        print("The list is empty.")
        return

    print("--- Rec List ---")
    for i, item in enumerate(input_list):
        # enumerate starts counting from 0 by default, so we add 1 for user-friendly numbering
        print(f"{i + 1}. {item}")
    print("---------------------\n")



# --- Example Usage ---
if __name__ == "__main__":
    # Sample user input (these should be consistent with how they appear in your book_list_features.txt)
    sample_user_books = [
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

    sample_user_ratings = [5, 5, 5, 3, 4, 5, 2, 4, 5, 4, 5, 4, 4, 5, 4, 5, 4]

    # sample_user_books = [
    # "the hobbit, or there and back again, j.r.r. tolkien",
    # "the girl on the train, paula hawkins",
    # "things fall apart, chinua achebe",
    # "anthem, ayn rand",
    # "the hunger games, suzanne collins",
    # "the maze runner, james dashner",
    # "the metamorphosis and other stories, franz kafka",
    # "the stranger, albert camus",
    # "into the water, paula hawkins",
    # "night, elie wiesel",
    # "a wrinkle in time, madeleine l'engle",
    # "shadow divers, robert kurson",
    # "silent spring, rachel carson",
    # "the great gatsby, f. scott fitzgerald",
    # "ways of seeing, john berger",
    # "1984, george orwell",
    # "all about love, bell hooks",
    # "raven's gate, anthony horowitz",
    # "touching spirit bear, ben mikaelsen",
    # "don quixote, miguel de cervantes saavedra",
    # "the elements, theodore gray",
    # "beowulf, unknown",
    # "the outsiders, s.e. hinton"
    # ]

    # sample_user_ratings = [4, 3, 5, 1, 4, 4, 5, 4, 2, 5, 2, 4, 4, 3, 5, 3, 4, 2, 3, 4, 4, 3, 4]


    # Define multiple settings to test
    genre_weights = [0.5, 0.8, 0] # Example genre weights
    num_features_settings = [600] # Example number of features
    user_reg_strengths = [5] # Example regularization strengths

    # Iterate through different combinations of settings
    for gw in genre_weights:
        for nf in num_features_settings:
            for urs in user_reg_strengths:
                print(f"\n#######################################################")
                print(f"### Running for GENRE_WEIGHT={gw}, NUM_FEATURES={nf}, NEW_USER_REGULARIZATION_STRENGTH={urs} ###")
                print(f"#######################################################")

                final_recommendations_df = get_combined_recommendations(
                    sample_user_books,
                    sample_user_ratings,
                    genre_weight=gw,
                    num_features=nf,
                    new_user_regularization_strength=urs
                )

                if final_recommendations_df is not None:
                    print(f"\n--- RECOMMENDATION STRATEGY 1: CB Top 200 Filtered by CF (output_limit=30) ---")
                    rec_strategy_2_list = recommend_cb_filtered_by_cf(final_recommendations_df, cb_initial_top_n=200, output_limit=30)
                    print_as_numbered_list(rec_strategy_2_list)

                    print(f"\n--- RECOMMENDATION STRATEGY 2: Scores Multiplied Together (output_limit=30) ---")
                    rec_strategy_3_list = recommend_by_multiplying_scores(final_recommendations_df, output_limit=30)
                    print_as_numbered_list(rec_strategy_3_list)

                    print(f"\n--- RECOMMENDATION STRATEGY 3: Pure CB (output_limit=30) ---")
                    rec_strategy_3_list = recommend_fraction_of_top_n(final_recommendations_df, cb_fraction= 1, n=30)
                    print_as_numbered_list(rec_strategy_3_list)
                
                else:
                    print(f"\n--- Recommendation generation failed for GW={gw}, NF={nf}, URS={urs} ---")


