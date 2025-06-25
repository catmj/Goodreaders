# Clustering books by keyword and genre features using k-modes (a k-means variant), using feature vectors of genres and (clustered) keywords.

# To run in a virtual environment, first do: pip install -r requirements.txt
# Importing necessary packages.
import pandas as pd
import numpy as np
from kmodes.kmodes import KModes

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
num_keywords = len(str_to_feat(feature_df.iloc[0,2])) # MAKE SURE COLUMNS MATCH.
num_genres = len(str_to_feat(feature_df.iloc[0,4])) # MAKE SURE COLUMNS MATCH.
num_features = num_keywords + num_genres
# Weight of keywords relative to genres.
keyword_weight = 0.8

# Isolating feature data into an array (keywords and genres).
# Creating the row for the feature vector of the first book.
data = np.hstack((np.array(str_to_feat(feature_df.iloc[0,2])),np.array(str_to_feat(feature_df.iloc[0,4])))) # MAKE SURE COLUMNS MATCH.
# Appending the rows for the feature vectors of each other book.
for i in range(1,num_rows):
    data = np.vstack((data,np.hstack((np.array(str_to_feat(feature_df.iloc[i,2])),np.array(str_to_feat(feature_df.iloc[i,4])))))) # MAKE SURE COLUMNS MATCH.

# Specifying number of clusters. Use "elbow_method.py" to determine optimal number.
num_clusters = 50
# Optimal is ~1/20 of number of books according to elbow chart. Probably less is better for our purposes.

# Running k-modes algorithm with custom dissimilarity metric.
kmode = KModes(n_clusters=num_clusters, init = "random", n_init = 5, max_iter = 20, verbose=1)
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
clusters = kmode.fit_predict(data)
# Optional prints for testing purposes.
# print("Cluster labels:", clusters)
# print("Cluster centroids:", kmode.cluster_centroids_)

# Adding a column of feature labels to the input file.
feature_df['cluster'] = clusters
feature_df.to_csv('books_clustered.csv', index=False)

# Print lists of each cluster to a CSV file.
cluster_list = []
for cluster in range(0,num_clusters):
    book_list = []
    for row in range(0,num_rows):
        if feature_df.iloc[row,6] == cluster: # MAKE SURE COLUMNS MATCH.
            # Consistent book formatting ('title, author') with other parts of the project.
            book_to_add = feature_df.iloc[row,0] + ", " + feature_df.iloc[row,1] # MAKE SURE COLUMNS MATCH.
            # Alternate book formatting.
            # book_to_add = []
            # book_to_add.append(feature_df.iloc[row,0])
            # book_to_add.append(feature_df.iloc[row,1])
            book_list.append(book_to_add)
    # Add the list of books (one cluster) to the list of clusters.
    cluster_list.append(book_list)
# Convert list of clusters to a dataframe.
output_df = pd.DataFrame({"titles_authors":cluster_list})
output_df.to_csv('books_by_cluster.csv', index=True)