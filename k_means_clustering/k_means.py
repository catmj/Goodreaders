# Clustering books by keyword and genre features using k-modes (a k-means variant).

# Importing necessary packages.
import pandas as pd
import numpy as np
import random
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
feature_df = pd.read_csv('books.csv')
num_rows = len(feature_df) # Do not count column names as a row.
num_features = len(str_to_feat(feature_df.iloc[1,2]))

# Isolating feature data into an array.
data = np.array(str_to_feat(feature_df.iloc[0,2]))
for i in range(1,num_rows):
    data = np.vstack((data,np.array(str_to_feat(feature_df.iloc[i,2]))))
# print(data)

# Specifying number of clusters.
num_clusters = 200
# clusters = {}




kmode = KModes(n_clusters=k, init = "random", n_init = 10, max_iter = 20, verbose=1)
kmode.fit_predict(data)




# Initializing random mode centroids.
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