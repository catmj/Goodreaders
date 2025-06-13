# Clustering books by keyword and genre features using k-modes (a k-means variant).
# Cosine similarity is not yet implemented.

# Importing necessary packages.
import pandas as pd
import numpy as np
import random
import csv
random.seed(38)
from kmodes.kmodes import KModes

# Defining Hamming distance.
def hamming_dist(vec_1, vec_2):
    distance = 0
    if len(vec_1) != len(vec_2):
        raise ValueError("Vectors must be of equal length.")
    for i in range(len(vec_1)):
        if vec_1[i] != vec_2[i]:
            distance += 1
    return int(distance)

# Defining cosine similarity.
def cosine_sim(vec_1, vec_2):
    if len(vec_1) != len(vec_2):
        raise ValueError("Vectors must be of equal length.")
    dot_product = np.dot(vec_1, vec_2)
    mag_vec1 = np.linalg.norm(vec_1)
    mag_vec2 = np.linalg.norm(vec_2)
    return dot_product / (mag_vec1 * mag_vec2)

# Defining a method that converts strings of feature vectors to actual feature vectors.
def str_to_feat(input):
    cleaned_input = input.strip("[").strip("]").split(", ")
    output = [int(x) for x in cleaned_input]
    return output

# Importing feature data.
feature_df = pd.read_csv('books_with_vectors.csv')
num_rows = len(feature_df) # Do not count column names as a row.
num_features = len(str_to_feat(feature_df.iloc[1,3])) # 2 for fake data, 3 for genre vectors.

# Isolating feature data into an array.
data = np.array(str_to_feat(feature_df.iloc[0,3])) # 2 for fake data, 3 for genre vectors.
for i in range(1,num_rows):
    data = np.vstack((data,np.array(str_to_feat(feature_df.iloc[i,3])))) # 2 for fake data, 3 for genre vectors.
# print(data)

# Specifying number of clusters. Use "elbow_method.py" to determine optimal number.
num_clusters = 1500
# Optimal is ~1/20 of number of books according to elbow chart. Probably less is better for our purposes.

# Running k-modes algorithm.
kmode = KModes(n_clusters=num_clusters, init = "random", n_init = 1, max_iter = 20, verbose=1)
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
        if feature_df.iloc[row,4] == cluster:
            # Consistent book formatting.
            book_to_add = feature_df.iloc[row,0] + ", " + feature_df.iloc[row,1]
            # Alternate book formatting.
            # book_to_add = []
            # book_to_add.append(feature_df.iloc[row,0])
            # book_to_add.append(feature_df.iloc[row,1])
            book_list.append(book_to_add)
    cluster_list.append(book_list)
output_df = pd.DataFrame({"titles_authors":cluster_list})
output_df.to_csv('books_by_cluster.csv', index=True)




# Initializing random mode centroids.
#clusters = {}
#initial_modes = random.sample(range(num_rows), num_clusters)
#for i in range(num_clusters):
#    center = initial_modes[i]
#    points = []
#    cluster = {
#        'center' : center,
#        'points' : []
#    }
#    clusters[i] = cluster
#print(clusters)