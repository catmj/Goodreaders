# Clustering books by keyword and genre features using k-means.
import pandas as pd

# Specifying number of clusters and importing feature data.
num_clusters = 1000
keyword_df = pd.read_csv('books.csv')