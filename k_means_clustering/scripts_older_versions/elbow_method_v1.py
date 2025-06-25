# Code for determining optimal number of clusters for k-modes, using feature vectors of genres and (clustered) keywords.

# To run in a virtual environment, first do: pip install -r requirements.txt
# Importing necessary packages.
import pandas as pd
import numpy as np
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt

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