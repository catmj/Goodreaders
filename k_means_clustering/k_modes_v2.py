# Clustering books by keyword and genre features using k-modes (a k-means variant).

# Importing necessary packages.
import pandas as pd
import numpy as np
import random
import csv
random.seed(38)
from kmodes.kmodes import KModes
from sklearn.metrics.pairwise import cosine_similarity
# from kmodes.util.dissim import matching_dissim, euclidean_dissim

# Important placeholders.
num_keywords = 0
num_genres = 0

# # Defining Hamming distance.
# def hamming_dist(vec_1, vec_2, **kw):
#     distance = 0
#     if len(vec_1) != len(vec_2):
#         raise ValueError("Vectors must be of equal length.")
#     for i in range(0,num_keywords): 
#         if vec_1[i] != vec_2[i]:
#             distance += keyword_weight
#     for i in range(0,num_genres): 
#         if vec_1[num_keywords+i] != vec_2[num_keywords+i]:
#             distance += 1
#     return distance

# Defining cosine similarity.
def cosine_dissim(X_array, Y_array, **kw):
    """
    Calculates 1 - cosine_similarity between two arrays of vectors.
    This function adheres to the signature expected by KModes.cat_dissim.

    Parameters:
    X_array : numpy.ndarray
        Shape (n_samples_X, n_features).
    Y_array : numpy.ndarray
        Shape (n_samples_Y, n_features).
    **kw : dict
        Additional keyword arguments (e.g., num_keywords, keyword_weight)
        if passed from KModes, though KModes doesn't typically forward custom kwargs here.
        For simplicity, we assume num_keywords and keyword_weight are global
        or handled otherwise.

    Returns:
    numpy.ndarray
        Dissimilarity matrix of shape (n_samples_X, n_samples_Y).
    """
    global num_keywords, keyword_weight # Declare global to access them.
    # Create copies to avoid modifying the original arrays in-place.
    X_weighted = X_array.astype(float).copy() # Ensure float type for multiplication.
    Y_weighted = Y_array.astype(float).copy() # Ensure float type for multiplication.
    # Apply weighting for keywords part of the vectors.
    # This operates on all rows/samples in a vectorized manner.
    if num_keywords > 0 and keyword_weight != 1.0: # Only apply if weighting is needed.
        X_weighted[:, :num_keywords] *= keyword_weight
        Y_weighted[:, :num_keywords] *= keyword_weight
    # Calculate cosine similarity matrix using sklearn's optimized function.
    # This handles zero vectors robustly (similarity will be 0 if one or both vectors are zero).
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
# Weight of keywords relative to genres.
keyword_weight = 0.8

# Isolating feature data into an array (keywords and genres).
# Ensure 'data' is float for cosine similarity.
data_list = []
for i in range(num_rows):
    row_data = np.hstack((np.array(str_to_feat(feature_df.iloc[i,2])),
                          np.array(str_to_feat(feature_df.iloc[i,4]))))
    data_list.append(row_data)
data = np.array(data_list, dtype=float) # Ensure data is float type for cosine similarity.

# Specifying number of clusters.
num_clusters = 300
# Optimal is ~1/20 of number of books according to elbow chart. Probably less is better for our purposes.

# Running k-modes algorithm with custom dissimilarity metric.
kmode = KModes(n_clusters=num_clusters, init = "random", n_init = 5, max_iter = 20, verbose=1)
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