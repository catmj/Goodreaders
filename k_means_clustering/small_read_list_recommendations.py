# Code for generating a list of recommendations for users who only submit 1-5 rated books using k-modes clusters.
# Google Gemini used to assist.

# To run in a virtual environment, first do: pip install -r requirements.txt
# Importing necessary packages.
import pandas as pd
import numpy as np
import ast

# Important global variable placeholders for Hamming distance and cosine dissimilarity.
num_keywords = 0
num_genres = 0
# Weight of keywords relative to genres.
keyword_weight = 0.8

def cosine_sim(vec1, vec2):
    # Convert lists to NumPy arrays for efficient computation.
    vec1 = np.array(vec1, dtype=float) # Ensure float type.
    vec2 = np.array(vec2, dtype=float) # Ensure float type.
    # Check if vectors have the same length.
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length to compute cosine similarity.")
    # Check num_keywords > 0 to prevent issues if num_keywords ends up being 0.
    if num_keywords > 0 and keyword_weight != 1.0:
        vec1_weighted = vec1.copy()
        vec2_weighted = vec2.copy()
        vec1_weighted[:num_keywords] *= keyword_weight
        vec2_weighted[:num_keywords] *= keyword_weight
        # Use vec1_weighted and vec2_weighted for dot product and norms.
        dot_product = np.dot(vec1_weighted, vec2_weighted)
        norm_vec1 = np.linalg.norm(vec1_weighted)
        norm_vec2 = np.linalg.norm(vec2_weighted)
    else:
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
    # Handle division by zero if either norm is zero.
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    else:
        similarity = dot_product / (norm_vec1 * norm_vec2)
        return similarity

# Defining a method that converts strings of feature vectors to actual feature vectors.
def str_to_feat(input):
    # Removing non-numerical characters.
    cleaned_input = input.strip("[").strip("]").split(", ")
    # Converting string to integers.
    output = [int(x) for x in cleaned_input]
    return output

# Sample user input (these should be consistent with how they appear in your book_list_features.txt).
sample_user_books = [
    "gideon the ninth, tamsyn muir",
    "flight of magpies, k.j. charles",
    "she who became the sun, shelley parker-chan",
    "boyfriend material, alexis hall",
    "summer sons, lee mandelo" # ,
    # "this is how you lose the time war, amal el-mohtar",
    # "the fifth season, n.k. jemisin",
    # "the king of attolia, megan whalen turner",
    # "iron widow, xiran jay zhao",
    # "all systems red, martha wells",
    # "the hobbit, or there and back again, j.r.r. tolkien",
    # "the girl on the train, paula hawkins",
    # "things fall apart, chinua achebe",
    # "anthem, ayn rand",
    # "the hunger games, suzanne collins"
]

# Get the length of the list of rated books.
num_ratings = len(sample_user_books)
# print(f"The number of ratings is: {num_ratings}")

# Importing feature data.
cluster_df = pd.read_csv('cluster_results_only_keywords/books_by_cluster_keywords_cosine_300_init1_ratings.csv')
feature_df = pd.read_csv('cluster_results_only_keywords/books_clustered_keywords_cosine_300_init1_ratings.csv')
# Various sizes of data.
num_clusters = len(cluster_df)
num_rows = len(feature_df) # Do not count column names as a row.
num_keywords = len(str_to_feat(feature_df.at[0,"keyword_vector"]))
num_genres = len(str_to_feat(feature_df.at[0,"genre_vector"]))
num_features = num_keywords + num_genres

# Finding the clusters for each book in the rated list.
clusters_to_use = []
clusters_to_use_unique = []
for rated_book in sample_user_books:
    for cluster in range(num_clusters):
        # Flag to check if the book was found.
        found_book_in_cluster = False 
        # Safely evaluate the string representation of the list.
        list_of_books_in_cluster_str = cluster_df.at[cluster, "titles_authors"]
        # Use ast.literal_eval to convert the string to a Python list.
        list_of_books_in_cluster = ast.literal_eval(list_of_books_in_cluster_str)
        for book_in_cluster in list_of_books_in_cluster:
            if rated_book == book_in_cluster:
                clusters_to_use.append(cluster)
                found_book_in_cluster = True
                break # Break from inner loop once book is found in a cluster.
        if found_book_in_cluster:
            break # Break from outer loop once book is found.
# Remove duplicate clusters.
for item in clusters_to_use:
    if item not in clusters_to_use_unique:
        clusters_to_use_unique.append(item)

# Remove books not in a cluster containing a rated book from the dataframe.
rows_to_remove = []
for row in range(num_rows):
    if feature_df.at[row, "cluster"] not in clusters_to_use_unique:
        rows_to_remove.append(row)
if rows_to_remove: # Only try to drop if there are rows to remove.
    feature_df.drop(rows_to_remove, inplace=True)
    # Reset the index to have a clean, sequential index after dropping rows.
    feature_df.reset_index(drop=True, inplace=True)

# Combine 'title' and 'author_name' into a new 'titles_authors' column in feature_df.
# Use .fillna('') to handle potential NaN values in title or author_name before combining.
feature_df['titles_authors'] = feature_df['title'].fillna('') + ', ' + feature_df['author_name'].fillna('')
# Strip any leading/trailing spaces from the combined string, if necessary.
feature_df['titles_authors'] = feature_df['titles_authors'].str.strip()

# Isolating feature data into an array (keywords and genres). Ensure 'data' is float for cosine similarity.
data_list = []
for i in range(len(feature_df)):
    row_data = np.hstack((np.array(str_to_feat(feature_df.at[i,"keyword_vector"])),np.array(str_to_feat(feature_df.at[i,"genre_vector"])))) # CHANGE BACK 
    data_list.append(row_data)
data = np.array(data_list, dtype=float) # Ensure data is float type for cosine similarity.

# Identifying feature vectors for rated books.
rated_books_features = [] # Initialize an empty list to hold the feature vectors
for rated_book_title_author in sample_user_books:
    # Find the row in feature_df that matches the rated book.
    # This efficiently filters the DataFrame to find the book.
    matching_rows = feature_df[feature_df['titles_authors'] == rated_book_title_author]
    if not matching_rows.empty:
        # Assuming there's only one match per book.
        book_row = matching_rows.iloc[0] # Get the first (and likely only) matching row.
        keyword_vec_str = book_row["keyword_vector"]
        genre_vec_str = book_row["genre_vector"]
        # Convert string representations of vectors (e.g., "[1, 0, 1]") into actual numerical lists/arrays using your str_to_feat function.
        keyword_feat = str_to_feat(keyword_vec_str)
        genre_feat = str_to_feat(genre_vec_str)
        # Concatenate the keyword and genre features into a single, comprehensive feature vector.
        full_feature_vector = np.hstack((np.array(keyword_feat), np.array(genre_feat)))
        rated_books_features.append(full_feature_vector) # Add the combined vector to the list.
    else:
        # This warning helps if a book from your sample_user_books couldn't be found in the filtered feature_df (e.g., if it wasn't in any of the relevant clusters).
        print(f"Warning: Rated book '{rated_book_title_author}' not found in the filtered feature_df.")
# Convert the list of feature vectors for rated books into a single NumPy array. This makes it easier to perform array operations later.
rated_books_features_array = np.array(rated_books_features, dtype=float)

# Identify and remove outliers using cosine dissimilarity.
sim_matrix = np.zeros((num_ratings,len(feature_df)))
for rating in range(num_ratings):
    rating_features = rated_books_features_array[rating,:]
    for book in range(len(feature_df)):
        book_features = data[book,:]
        sim_matrix[rating,book] = cosine_sim(rating_features,book_features)
sim_averages = np.mean(sim_matrix, axis=0)
mean_value = np.mean(sim_averages)
std_dev_value = np.std(sim_averages)
# Specify the number of standard deviations to be considered an outlier. Should be positive.
std_num_outlier = 2.0
# Create list of outlier rows to remove.
outliers_to_remove = []
for book_idx in range(len(sim_averages)): # Loop through the sim_averages, which corresponds to current feature_df indices.
    # We consider books with average similarity significantly below the mean as outliers.
    if sim_averages[book_idx] < (mean_value - std_num_outlier * std_dev_value):
        outliers_to_remove.append(book_idx)
# Remove identified outlier rows from the dataframe.
if outliers_to_remove: # Only try to drop if there are outliers to remove
    # Use the actual index labels of the DataFrame, which are the current row numbers.
    feature_df.drop(feature_df.index[outliers_to_remove], inplace=True)
    # Reset the index again to maintain a clean sequential index
    feature_df.reset_index(drop=True, inplace=True)
    print(f"Removed {len(outliers_to_remove)} outlier books.")
else:
    print("No outliers found to remove based on the specified criteria.")

# After removing outliers, 'data' array needs to be regenerated to reflect the current state of 'feature_df'.
# This is crucial for any subsequent operations that rely on 'data' being aligned with 'feature_df'.
# data_list_after_outlier_removal = []
# for i in range(len(feature_df)):
#     row_data_after_removal = np.hstack((np.array(str_to_feat(feature_df.at[i,"keyword_vector"])),np.array(str_to_feat(feature_df.at[i,"genre_vector"]))))
#     data_list_after_outlier_removal.append(row_data_after_removal)
# data = np.array(data_list_after_outlier_removal, dtype=float)

# Keep the top N most popular books, and then keep the top M highest rated books. Vice versa if N<M (worse results).
highest_hits_to_keep = round(len(feature_df)*0.9) # N
highest_ratings_to_keep = min(50,highest_hits_to_keep) # M
if highest_hits_to_keep >= highest_ratings_to_keep:
    # Keep the top N most popular books.
    sorted_hits_feature_df = feature_df.sort_values(by='num_ratings', ascending=False)
    sorted_hits_feature_df.reset_index(drop=True, inplace=True)
    low_hits_to_drop = range(highest_hits_to_keep,len(sorted_hits_feature_df))
    sorted_hits_feature_df.drop(low_hits_to_drop, inplace=True)
    # Keep the top M highest rated books.
    final_feature_df = sorted_hits_feature_df.sort_values(by='rating', ascending=False)
    final_feature_df.reset_index(drop=True, inplace=True)
    low_ratings_to_drop = range(highest_ratings_to_keep,len(final_feature_df))
    final_feature_df.drop(low_ratings_to_drop, inplace=True)
if highest_hits_to_keep < highest_ratings_to_keep:
    # Keep the top M highest rated books.
    sorted_ratings_feature_df = feature_df.sort_values(by='rating', ascending=False)
    sorted_ratings_feature_df.reset_index(drop=True, inplace=True)
    low_ratings_to_drop = range(highest_ratings_to_keep,len(sorted_ratings_feature_df))
    sorted_ratings_feature_df.drop(low_ratings_to_drop, inplace=True)
    # Keep the top N most popular books.
    final_feature_df = sorted_ratings_feature_df.sort_values(by='num_ratings', ascending=False)
    final_feature_df.reset_index(drop=True, inplace=True)
    low_hits_to_drop = range(highest_hits_to_keep,len(final_feature_df))
    final_feature_df.drop(low_hits_to_drop, inplace=True)

# Results and testing.
# for index, row in final_feature_df.iterrows():
#     print(f"- Title: {row['title']}, Author: {row['author_name']}")
final_feature_df.to_csv('recommendations.csv', index=True)