# Code for generating a list of recommendations for users who only submit 1-5 rated books using k-modes clusters.
# Google Gemini used to assist.

# To run in a virtual environment, first do: pip install -r requirements.txt
# Importing necessary packages.
import pandas as pd
import numpy as np
import ast
import os # Import os for path manipulation
import sys # Import sys for potential exit on error

# Important global variable placeholders for Hamming distance and cosine dissimilarity.
# These will be set once when the module is loaded by reading the feature data.
num_keywords = 0
num_genres = 0
# Weight of keywords relative to genres.
keyword_weight = 0.8

# --- Data Loading (Global, runs once when module is imported) ---
# Determine the absolute path to the data files.
# The script `small_read_list_recommendations_v2.py` is assumed to be in 'Goodreaders/k_means_clustering/'.
# The data is in 'Goodreaders/k_modes_clustering/cluster_results_both_cosine_dissim/'.
# So, from the current script's directory, we go up one level (to 'Goodreaders'),
# then down into 'k_modes_clustering', and then into 'cluster_results_both_cosine_dissim'.
current_script_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir, 'k_means_clustering', 'cluster_results_both_cosine_dissim'))

# File paths for the dataframes.
cluster_df_path = os.path.join(data_dir, 'books_by_cluster_both_cosine_300_init1_ratings.csv')
feature_df_path = os.path.join(data_dir, 'books_clustered_both_cosine_300_init1_ratings.csv')

# Load dataframes globally to avoid reloading them every time the function is called.
try:
    cluster_df_full = pd.read_csv(cluster_df_path)
    feature_df_full = pd.read_csv(feature_df_path)

    # NEW: Generate 'titles_authors' column for the global feature_df_full
    feature_df_full['titles_authors'] = feature_df_full['title'].fillna('') + ', ' + feature_df_full['author_name'].fillna('')
    feature_df_full['titles_authors'] = feature_df_full['titles_authors'].str.strip()

    # Set global variables based on the loaded feature_df_full for vector dimensions.
    # Use ast.literal_eval for safe string conversion to list/array.
    temp_first_keyword_vec = ast.literal_eval(feature_df_full.at[0,"keyword_vector"])
    temp_first_genre_vec = ast.literal_eval(feature_df_full.at[0,"genre_vector"])
    num_keywords = len(temp_first_keyword_vec)
    num_genres = len(temp_first_genre_vec)
except FileNotFoundError as e:
    # Print error and exit if critical data files are missing.
    print(f"Error: Data file not found for K-Means recommender. Please ensure files are in {data_dir}. Error: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    # Catch any other exceptions during data loading.
    print(f"An unexpected error occurred during K-Means data loading: {e}", file=sys.stderr)
    sys.exit(1)


def cosine_sim(vec1, vec2):
    """
    Computes the cosine similarity between two vectors, with optional weighting for keywords.
    """
    vec1 = np.array(vec1, dtype=float)
    vec2 = np.array(vec2, dtype=float)

    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length to compute cosine similarity.")

    # Apply keyword weighting if applicable and num_keywords is set.
    if num_keywords > 0 and keyword_weight != 1.0:
        vec1_weighted = vec1.copy()
        vec2_weighted = vec2.copy()
        vec1_weighted[:num_keywords] *= keyword_weight
        vec2_weighted[:num_keywords] *= keyword_weight
        dot_product = np.dot(vec1_weighted, vec2_weighted)
        norm_vec1 = np.linalg.norm(vec1_weighted)
        norm_vec2 = np.linalg.norm(vec2_weighted)
    else:
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

    # Handle division by zero.
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    else:
        similarity = dot_product / (norm_vec1 * norm_vec2)
        return similarity

def str_to_feat(input_str: str) -> list:
    """
    Converts a string representation of a list (e.g., "[1, 0, 1]") into an actual Python list of integers.
    """
    try:
        return ast.literal_eval(input_str)
    except (ValueError, SyntaxError) as e:
        print(f"Error converting string to feature vector: '{input_str}'. Error: {e}", file=sys.stderr)
        return [] # Return an empty list on error.


def get_kmeans_recommendations(sample_user_books: list, maximum_books: int = 20) -> list:
    """
    Generates book recommendations for users with a small read list using K-Means clustering.
    It identifies clusters relevant to the user's read books, filters out outliers,
    and returns a list of top popular and highly-rated books from those clusters,
    excluding books already read by the user.

    Args:
        sample_user_books (list): A list of book titles (e.g., "book title, author name")
                                  that the user has read. These titles must exactly match
                                  the format in the underlying data.
        maximum_books (int): The maximum number of recommendations to return.

    Returns:
        list: A list of recommended book titles (e.g., "book title, author name").
              Returns an empty list if no recommendations can be made.
    """
    if not sample_user_books:
        return []

    # Create working copies of the dataframes to avoid modifying the global ones.
    cluster_df = cluster_df_full.copy()
    feature_df = feature_df_full.copy()

    # Find the clusters associated with the user's read books.
    clusters_to_use = []
    for rated_book in sample_user_books:
        # Iterate through clusters to find which one contains the rated book.
        for cluster_idx in range(len(cluster_df)):
            list_of_books_in_cluster = ast.literal_eval(cluster_df.at[cluster_idx, "titles_authors"])
            if rated_book in list_of_books_in_cluster:
                clusters_to_use.append(cluster_idx)
                break # Move to the next rated book once found

    # Get unique clusters.
    clusters_to_use_unique = list(set(clusters_to_use))
    
    if not clusters_to_use_unique:
        return []

    # Filter feature_df to include only books from the identified clusters.
    feature_df = feature_df[feature_df['cluster'].isin(clusters_to_use_unique)].copy()
    feature_df.reset_index(drop=True, inplace=True)
    
    if feature_df.empty:
        return []

    # The 'titles_authors' column is already in feature_df_full, and it's copied to feature_df
    # when feature_df_full is copied. So, the explicit creation here for feature_df is redundant
    # but harmless. I'll keep it for clarity that this column is expected.
    feature_df['titles_authors'] = feature_df['title'].fillna('') + ', ' + feature_df['author_name'].fillna('')
    feature_df['titles_authors'] = feature_df['titles_authors'].str.strip()


    # Prepare feature vectors for all books in the filtered feature_df.
    data_list = []
    for i in range(len(feature_df)):
        keyword_feat = str_to_feat(feature_df.at[i,"keyword_vector"])
        genre_feat = str_to_feat(feature_df.at[i,"genre_vector"])
        row_data = np.hstack((np.array(keyword_feat), np.array(genre_feat)))
        data_list.append(row_data)
    data = np.array(data_list, dtype=float)

    # Identify feature vectors for the actual books provided by the user.
    rated_books_features = []
    # Fetch from the full feature_df to ensure we get features for all user books,
    # even if they weren't in the selected clusters (though ideally they should be).
    for rated_book_title_author in sample_user_books:
        matching_rows = feature_df_full[feature_df_full['titles_authors'] == rated_book_title_author]
        if not matching_rows.empty:
            book_row = matching_rows.iloc[0]
            keyword_feat = str_to_feat(book_row["keyword_vector"])
            genre_feat = str_to_feat(book_row["genre_vector"])
            full_feature_vector = np.hstack((np.array(keyword_feat), np.array(genre_feat)))
            rated_books_features.append(full_feature_vector)

    if not rated_books_features:
        return []

    rated_books_features_array = np.array(rated_books_features, dtype=float)
    num_processed_ratings = rated_books_features_array.shape[0]

    # Calculate similarity matrix for outlier identification.
    sim_matrix = np.zeros((num_processed_ratings, len(feature_df)))
    for rating_idx in range(num_processed_ratings):
        rating_features = rated_books_features_array[rating_idx,:]
        for book_idx in range(len(feature_df)):
            book_features = data[book_idx,:]
            sim_matrix[rating_idx,book_idx] = cosine_sim(rating_features,book_features)
            
    sim_averages = np.mean(sim_matrix, axis=0)

    # Remove outliers based on similarity averages.
    if not np.all(sim_averages == sim_averages[0]): # Check if all similarities are identical
        mean_value = np.mean(sim_averages)
        std_dev_value = np.std(sim_averages)
        std_num_outlier = 2.0
        outliers_to_remove = []
        for book_idx in range(len(sim_averages)):
            if sim_averages[book_idx] < (mean_value - std_num_outlier * std_dev_value):
                outliers_to_remove.append(book_idx)
        
        if outliers_to_remove:
            feature_df.drop(feature_df.index[outliers_to_remove], inplace=True)
            feature_df.reset_index(drop=True, inplace=True)

    # Remove books the user has already read from the recommendation pool.
    feature_df = feature_df[~feature_df['titles_authors'].isin(sample_user_books)].copy()
    if feature_df.empty:
        return []
        
    # Sort and filter for the top 'maximum_books' recommendations.
    # Prioritize by number of ratings (popularity) then by average rating.
    final_feature_df = feature_df.sort_values(by=['num_ratings', 'rating'], ascending=[False, False])
    
    # Get the top 'maximum_books' titles.
    recommended_titles = final_feature_df['titles_authors'].head(maximum_books).tolist()

    return recommended_titles

# This section is for standalone testing of the function.
# It will not run when this file is imported as a module.
if __name__ == "__main__":
    test_user_books = [
        "lot lizards, ray garton",
        "gideon the ninth, tamsyn muir",
        "the hobbit, or there and back again, j.r.r. tolkien"
    ]
    test_maximum_books = 10
    
    print("Generating K-Means recommendations...")
    recommendations = get_kmeans_recommendations(test_user_books, test_maximum_books)
    
    if recommendations:
        print("\nK-Means Recommendations:")
        for rec in recommendations:
            print(f"- {rec}")
    else:
        print("No K-Means recommendations could be generated with the given input.")