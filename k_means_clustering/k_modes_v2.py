# Clustering books by keyword and genre features using k-modes (a k-means variant).

# Importing necessary packages.
import pandas as pd
import numpy as np
import random
import csv
random.seed(38)
from kmodes.kmodes import KModes
from sklearn.metrics.pairwise import cosine_similarity
from kmodes.util.dissim import matching_dissim, euclidean_dissim

# Important placeholders.
num_keywords = 0
num_genres = 0
# Weight of keywords relative to genres.
keyword_weight = 0.8

# Defining Hamming distance.
def hamming_dist(X_array, Y_array, **kw):
    """
    Calculates a weighted Hamming dissimilarity matrix between two arrays of vectors. This function adheres to the signature expected by KModes.cat_dissim.
    Parameters:
    X_array : numpy.ndarray
        First array of data points. Shape (n_samples_X, n_features).
    Y_array : numpy.ndarray
        Second array of data points (e.g., cluster centroids or other data points). Shape (n_samples_Y, n_features).
    **kw : dict
        Additional keyword arguments (currently not used within this function as num_keywords, num_genres, and keyword_weight are assumed global).
    Returns:
    numpy.ndarray
        Dissimilarity matrix of shape (n_samples_X, n_samples_Y), where each element (i, j) is the weighted Hamming distance between X_array[i] and Y_array[j].
    """
    # Declare globals to ensure the function uses the current values of these variables defined outside the function.
    global num_keywords, num_genres, keyword_weight
    # Ensure arrays are float type for proper weighted sum, as keyword_weight is float.
    X_processed = X_array.astype(float)
    Y_processed = Y_array.astype(float)
    n_features = X_processed.shape[1] # Total number of features (keywords + genres).
    # Initialize the dissimilarity matrix with zeros.
    # This matrix will store the weighted Hamming distance between each row in X_processed and each row in Y_processed.
    dissimilarity_matrix = np.zeros((X_processed.shape[0], Y_processed.shape[0]))
    # Iterate over each feature (column) to calculate mismatches and apply weights.
    for col_idx in range(n_features):
        # Create broadcastable views for the current column from both arrays. X_col will be (n_samples_X, 1) and Y_col will be (1, n_samples_Y).
        # This allows a direct comparison that results in a (n_samples_X, n_samples_Y). Boolean matrix indicating mismatches for this specific column.
        mismatches = (X_processed[:, col_idx][:, np.newaxis] != Y_processed[:, col_idx][np.newaxis, :])
        # Apply the appropriate weight based on whether the feature is a keyword or a genre.
        if col_idx < num_keywords:
            # If it's a keyword feature, add mismatches multiplied by keyword_weight.
            dissimilarity_matrix += mismatches * keyword_weight
        else:
            # If it's a genre feature, add mismatches multiplied by 1 (default weight).
            dissimilarity_matrix += mismatches * 1.0 # Use 1.0 to ensure float arithmetic.
    return dissimilarity_matrix

# Defining cosine similarity.
def cosine_dissim(X_array, Y_array, **kw):
    """
    Calculates 1-cosine_similarity between two arrays of vectors. This function adheres to the signature expected by KModes.cat_dissim.
    Parameters:
    X_array : numpy.ndarray
        Shape (n_samples_X, n_features).
    Y_array : numpy.ndarray
        Shape (n_samples_Y, n_features).
    **kw : dict
        Additional keyword arguments (e.g., num_keywords, keyword_weight) if passed from KModes, though KModes doesn't typically forward custom kwargs here.
        For simplicity, we assume num_keywords and keyword_weight are global or handled otherwise.
    Returns:
    numpy.ndarray
        Dissimilarity matrix of shape (n_samples_X, n_samples_Y).
    """
    global num_keywords, keyword_weight # Declare global to access them.
    # Create copies to avoid modifying the original arrays in-place.
    X_weighted = X_array.astype(float).copy() # Ensure float type for multiplication.
    Y_weighted = Y_array.astype(float).copy() # Ensure float type for multiplication.
    # Apply weighting for keywords part of the vectors. This operates on all rows/samples in a vectorized manner.
    if num_keywords > 0 and keyword_weight != 1.0: # Only apply if weighting is needed.
        X_weighted[:, :num_keywords] *= keyword_weight
        Y_weighted[:, :num_keywords] *= keyword_weight
    # Calculate cosine similarity matrix using sklearn's optimized function. This handles zero vectors robustly (similarity will be 0 if one or both vectors are zero).
    similarity_matrix = cosine_similarity(X_weighted, Y_weighted)
    # Convert similarity to dissimilarity (1 - similarity).
    dissimilarity_matrix = 1 - similarity_matrix
    return dissimilarity_matrix

# Defining a method that converts strings of feature vectors to actual feature vectors.
def str_to_feat(input):
    cleaned_input = input.strip("[").strip("]").split(", ")
    output = [int(x) for x in cleaned_input]
    return output

# Importing feature data.
feature_df = pd.read_csv('books_with_vectors_big.csv')
# Various sizes of data.
num_rows = len(feature_df) # Do not count column names as a row.
num_keywords = len(str_to_feat(feature_df.iloc[0,2])) # MAKE SURE COLUMNS MATCH.
num_genres = len(str_to_feat(feature_df.iloc[0,4])) # MAKE SURE COLUMNS MATCH.
num_features = num_keywords + num_genres

# Isolating feature data into an array (keywords and genres).
# Ensure 'data' is float for cosine similarity.
data_list = []
for i in range(num_rows):
    row_data = np.hstack((np.array(str_to_feat(feature_df.iloc[i,2])),
                          np.array(str_to_feat(feature_df.iloc[i,4]))))
    data_list.append(row_data)
data = np.array(data_list, dtype=float) # Ensure data is float type for cosine similarity.

# Specifying number of clusters.
num_clusters = 200
# Optimal is ~1/20 of number of books according to elbow chart. Probably less is better for our purposes.

# Running k-modes algorithm with custom dissimilarity metric (hamming_dist, cosine_dissim, matching_dissim, euclidean_dissim).
kmode = KModes(n_clusters=num_clusters, init = "random", n_init = 1, max_iter = 20, verbose=1, cat_dissim=cosine_dissim)
clusters = kmode.fit_predict(data)
# print("Cluster labels:", clusters)
# print("Cluster centroids:", kmode.cluster_centroids_)

# Adding a column of feature labels to the input file.
feature_df['cluster'] = clusters
feature_df.to_csv('books_clustered.csv', index=False)

# Print lists of each cluster.
cluster_list = []
for cluster in range(0,num_clusters):
    book_list = []
    for row in range(0,num_rows):
        if feature_df.iloc[row,6] == cluster: # MAKE SURE COLUMNS MATCH.
            # Consistent book formatting.
            book_to_add = feature_df.iloc[row,0] + ", " + feature_df.iloc[row,1] # MAKE SURE COLUMNS MATCH.
            # Alternate book formatting.
            # book_to_add = []
            # book_to_add.append(feature_df.iloc[row,0])
            # book_to_add.append(feature_df.iloc[row,1])
            book_list.append(book_to_add)
    cluster_list.append(book_list)
output_df = pd.DataFrame({"titles_authors":cluster_list})
output_df.to_csv('books_by_cluster.csv', index=True)