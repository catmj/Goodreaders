# Code for determining optimal number of clusters for k-modes.

# Importing necessary packages.
import pandas as pd
import numpy as np
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt

# Defining a method that converts strings of feature vectors to actual feature vectors.
def str_to_feat(input):
    cleaned_input = input.strip("[").strip("]").split(", ")
    output = [int(x) for x in cleaned_input]
    return output

# Importing feature data.
feature_df = pd.read_csv('books.csv')
num_rows = len(feature_df) # Do not count column names as a row.
num_features = len(str_to_feat(feature_df.iloc[0,2]))

# Isolating feature data into an array.
data = np.vstack((np.array(str_to_feat(feature_df.iloc[0,2])),np.array(str_to_feat(feature_df.iloc[1,2]))))
for i in range(2,num_rows):
    data = np.vstack((data,np.array(str_to_feat(feature_df.iloc[i,2]))))

# Set numbers of clusters to test.
K = []
K_num_points = 50
K_spacing = 20
for i in range(1,K_num_points+1):
    k = i*K_spacing
    K.append(k)

# Determine costs.
cost = []
for k in list(K):
    kmode = KModes(n_clusters=k, init = "random", n_init = 10, max_iter = 20, verbose=1)
    kmode.fit_predict(data)
    cost.append(kmode.cost_)

# Create elbow plot.
plt.plot(K, cost, 'x-')
plt.xlabel('Number of Clusters')
plt.ylabel('Cost')
plt.title('Elbow Curve')
plt.show()