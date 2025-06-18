# Code for determining optimal number of clusters for k-modes, using feature vectors of genres and (clustered) keywords.

# To run first do: pip install -r requirements.txt
# Importing necessary packages.
import pandas as pd
import numpy as np
# import random
# import csv
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
# Setting random seed.
# random.seed(38)

# Defining a method that converts strings of feature vectors to actual feature vectors.
def str_to_feat(input):
    # Removing non-numerical characters.
    cleaned_input = input.strip("[").strip("]").split(", ")
    # Converting string to integers.
    output = [int(x) for x in cleaned_input]
    return output

# Importing feature data.
feature_df = pd.read_csv('books_test_nonrandom.csv')
# Various sizes of data.
num_rows = len(feature_df) # Do not count column names as a row.
num_keywords = len(str_to_feat(feature_df.iloc[0,2])) # MAKE SURE COLUMNS MATCH.
num_genres = len(str_to_feat(feature_df.iloc[0,4])) # MAKE SURE COLUMNS MATCH.
num_features = num_keywords + num_genres

# Isolating feature data into an array (keywords and genres).
# Creating the row for the feature vector of the first book.
data = np.hstack((np.array(str_to_feat(feature_df.iloc[0,2])),np.array(str_to_feat(feature_df.iloc[0,4])))) # MAKE SURE COLUMNS MATCH.
# Appending the rows for the feature vectors of each other book.
for i in range(1,num_rows):
    data = np.vstack((data,np.hstack((np.array(str_to_feat(feature_df.iloc[i,2])),np.array(str_to_feat(feature_df.iloc[i,4])))))) # MAKE SURE COLUMNS MATCH.

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
    # Running k-modes algorithm with unweighted Hamming distance as the dissimilarity metric.
    kmode = KModes(n_clusters=k, init = "random", n_init = 5, max_iter = 20, verbose=1)
    kmode.fit_predict(data)
    # Collecting costs for use in elbow plot.
    cost.append(kmode.cost_)

# Create elbow plot. The bend in the plot is approximately the optimal number of clusters in terms of costs.
plt.plot(K, cost, 'x-')
plt.xlabel('Number of Clusters')
plt.ylabel('Cost')
plt.title('Elbow Curve')
plt.show()