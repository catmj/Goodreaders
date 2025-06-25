# Code for determining optimal number of clusters for k-modes, using custom dissimilarity metrics with different weights given to keywords and genres.
# Google Gemini used to assist.

# To run in a virtual environment, first do: pip install -r requirements.txt
# Importing necessary packages.
import pandas as pd
import numpy as np
from kmodes.kmodes import KModes
from kmodes.util.dissim import matching_dissim, euclidean_dissim
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Important global variable placeholders for Hamming distance and cosine dissimilarity.
num_keywords = 0
num_genres = 0
# Weight of keywords relative to genres.
keyword_weight = 0.8

# Defining weighted Hamming distance.
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
    # Reshape 1D inputs to 2D (single row, n_features columns) for consistent slicing.
    if X_processed.ndim == 1:
        X_processed = X_processed.reshape(1, -1) # Reshape to (1, n_features).
    if Y_processed.ndim == 1:
        Y_processed = Y_processed.reshape(1, -1) # Reshape to (1, n_features).
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

# Defining weighted cosine dissimilarity.
def cosine_dissim(X_array, Y_array, **kw):
    """
    Calculates 1-cosine_similarity between two arrays of vectors.
    This function now handles both 1D and 2D input arrays by reshaping them to 2D for the vectorized operations.
    Parameters:
    X_array : numpy.ndarray
        First array of data points. Can be shape (n_features,) for a single point or (n_samples_X, n_features) for multiple points.
    Y_array : numpy.ndarray
        Second array of data points. Can be shape (n_features,) for a single point or (n_samples_Y, n_features) for multiple points (e.g., cluster centroids).
    **kw : dict
        Additional keyword arguments. (Currently not explicitly used in logic, as num_keywords and keyword_weight are assumed global).
    Returns:
    numpy.ndarray
        Dissimilarity matrix of shape (n_samples_X_effective, n_samples_Y_effective). If input was 1D, the effective sample size is 1.
    """
    global num_keywords, keyword_weight # Declare global to access them.
    # Ensure float type for multiplication and create copies to avoid modifying originals.
    X_processed = X_array.astype(float).copy()
    Y_processed = Y_array.astype(float).copy()
    # Reshape 1D inputs to 2D (single row, n_features columns) for consistent slicing and compatibility with sklearn.metrics.pairwise.cosine_similarity.
    if X_processed.ndim == 1:
        X_processed = X_processed.reshape(1, -1) # Reshape to (1, n_features).
    if Y_processed.ndim == 1:
        Y_processed = Y_processed.reshape(1, -1) # Reshape to (1, n_features).
    # Apply weighting for keywords part of the vectors. This operates on all rows/samples in a vectorized manner.
    # Check num_keywords > 0 to prevent issues if num_keywords ends up being 0.
    if num_keywords > 0 and keyword_weight != 1.0:
        X_processed[:, :num_keywords] *= keyword_weight
        Y_processed[:, :num_keywords] *= keyword_weight
    # Calculate cosine similarity matrix using sklearn's optimized function. This handles zero vectors robustly (similarity will be 0 if one or both vectors are zero).
    similarity_matrix = cosine_similarity(X_processed, Y_processed)
    # Convert similarity to dissimilarity (1 - similarity).
    dissimilarity_matrix = 1 - similarity_matrix
    return dissimilarity_matrix

# Defining a method that converts strings of feature vectors to actual feature vectors.
def str_to_feat(input):
    # Removing non-numerical characters.
    cleaned_input = input.strip("[").strip("]").split(", ")
    # Converting string to integers.
    output = [int(x) for x in cleaned_input]
    return output

# Importing feature data.
feature_df = pd.read_csv('books_with_vectors_and_sums.csv')
# Various sizes of data.
num_rows = len(feature_df) # Do not count column names as a row.
num_keywords = len(str_to_feat(feature_df.at[0,"keyword_vector"]))
num_genres = len(str_to_feat(feature_df.at[0,"genre_vector"]))
num_features = num_keywords + num_genres

# Isolating feature data into an array (keywords and genres). Ensure 'data' is float for cosine similarity.
data_list = []
for i in range(num_rows):
    row_data = np.hstack((np.array(str_to_feat(feature_df.at[i,"keyword_vector"])),np.array(str_to_feat(feature_df.at[i,"genre_vector"]))))
    data_list.append(row_data)
data = np.array(data_list, dtype=float) # Ensure data is float type for cosine similarity.

# Set numbers of clusters to test. Can be heavily modified to return any desired list of integers K.
K = []
K_num_points = 25
K_spacing = 10
for i in range(1,K_num_points+1):
    k = i*K_spacing
    K.append(k)

# Determine costs for each number of clusters tested.
cost = []
for k in list(K):
    # Running k-modes algorithm with custom dissimilarity metric (hamming_dist, cosine_dissim, matching_dissim, euclidean_dissim).
    kmode = KModes(n_clusters=k, init = "random", n_init = 5, max_iter = 20, verbose=1, cat_dissim=hamming_dist)
    """
    Initializes a K-Modes clustering model.
    Parameters:
        n_clusters : int 
            The number of clusters to form, the 'k' in K-Modes.
        init : str 
            Method for initialization. "random" selects random centroids from the data. "Huang" and "Cao" are other options.
        n_init : int 
            Number of times the K-Modes algorithm will be run with different centroid seeds. The final result will be the best output of n_init consecutive runs in terms of cost.
        max_iter : int 
            Maximum number of iterations of the K-Modes algorithm for a single run.
        verbose : int 
            Verbosity mode. 0 = silent, 1 = progress messages.
        cat_dissim : callable 
            Function to compute the dissimilarity between two categorical vectors.
    Returns:
        KModes
            An initialized K-Modes clustering model object.
    """
    # Get cluster indices from KModes.
    kmode.fit_predict(data)
    # Collecting costs for use in elbow plot.
    cost.append(kmode.cost_)

# Create elbow plot. The bend in the plot is approximately the optimal number of clusters in terms of costs.
plt.plot(K, cost, 'x-')
plt.xlabel('Number of Clusters')
plt.ylabel('Cost')
plt.title('Elbow Curve')
plt.show()