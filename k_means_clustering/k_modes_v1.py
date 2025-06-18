# Clustering books by keyword and genre features using k-modes (a k-means variant).

# To run first do: pip install -r requirements.txt
# Importing necessary packages.
import pandas as pd
import numpy as np
# import random
# import csv
from kmodes.kmodes import KModes
# Setting random seed.
# random.seed(38)

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
data = np.hstack((np.array(str_to_feat(feature_df.iloc[0,2])),np.array(str_to_feat(feature_df.iloc[0,4])))) # MAKE SURE COLUMNS MATCH.
for i in range(1,num_rows):
    data = np.vstack((data,np.hstack((np.array(str_to_feat(feature_df.iloc[i,2])),np.array(str_to_feat(feature_df.iloc[i,4])))))) # MAKE SURE COLUMNS MATCH.

# Specifying number of clusters. Use "elbow_method.py" to determine optimal number.
num_clusters = 50
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