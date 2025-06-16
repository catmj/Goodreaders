# Code for determining optimal number of clusters for k-modes.

# Importing necessary packages.
import pandas as pd
import numpy as np
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
from kmodes.util.dissim import matching_dissim, euclidean_dissim

# Defining a method that converts strings of feature vectors to actual feature vectors.
def str_to_feat(input):
    cleaned_input = input.strip("[").strip("]").split(", ")
    output = [int(x) for x in cleaned_input]
    return output

# Importing feature data.
feature_df = pd.read_csv('books_test_nonrandom.csv')
num_rows = len(feature_df) # Do not count column names as a row.
num_features = len(str_to_feat(feature_df.iloc[0,3])) # 2 for fake data, 3 for genre vectors.

# Isolating feature data into an array.
data = np.vstack((np.array(str_to_feat(feature_df.iloc[0,3])),np.array(str_to_feat(feature_df.iloc[1,3])))) # 2 for fake data, 3 for genre vectors.
for i in range(2,num_rows):
    data = np.vstack((data,np.array(str_to_feat(feature_df.iloc[i,3])))) # 2 for fake data, 3 for genre vectors.

# Set numbers of clusters to test.
K = []
K_num_points = 25
K_spacing = 10
for i in range(1,K_num_points+1):
    k = i*K_spacing
    K.append(k)

# Determine costs.
cost = []
for k in list(K):
    kmode = KModes(n_clusters=k, init = "random", n_init = 5, max_iter = 20, verbose=1)
    kmode.fit_predict(data)
    cost.append(kmode.cost_)

# Create elbow plot.
plt.plot(K, cost, 'x-')
plt.xlabel('Number of Clusters')
plt.ylabel('Cost')
plt.title('Elbow Curve')
plt.show()