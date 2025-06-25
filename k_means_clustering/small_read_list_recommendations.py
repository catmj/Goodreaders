# Code for generating a list of recommendations for users who only submit 1-5 rated books using k-modes clusters.

# To run in a virtual environment, first do: pip install -r requirements.txt
# Importing necessary packages.
import pandas as pd
import ast

# Defining a method that converts strings of feature vectors to actual feature vectors.
def str_to_feat(input):
    # Removing non-numerical characters.
    cleaned_input = input.strip("[").strip("]").split(", ")
    # Converting string to integers.
    output = [int(x) for x in cleaned_input]
    return output

# Sample user input (these should be consistent with how they appear in your book_list_features.txt).
sample_user_books = [
    "gideon the ninth, tamsyn muir",
    "the fifth season, n.k. jemisin",
    "the king of attolia, megan whalen turner",
    "iron widow, xiran jay zhao",
    "all systems red, martha wells"
]
sample_user_ratings = [5, 5, 5, 3, 4]
# sample_user_books = [
#     "the hobbit, or there and back again, j.r.r. tolkien",
#     "the girl on the train, paula hawkins",
#     "things fall apart, chinua achebe",
#     "anthem, ayn rand",
#     "the hunger games, suzanne collins"
#     ]
# sample_user_ratings = [4, 3, 5, 1, 4]

# Get the length of the list of rated books.
if len(sample_user_books) != len(sample_user_ratings):
    raise ValueError("Error: The number of books does not match the number of ratings.")
else:
    num_ratings = len(sample_user_ratings)
    # print(f"The number of ratings is: {num_ratings}")

# Importing feature data.
cluster_df = pd.read_csv('cluster_results_both_cosine_dissim/books_by_cluster_both_cosine_300_init1_ratings.csv')
feature_df = pd.read_csv('cluster_results_both_cosine_dissim/books_clustered_both_cosine_300_init1_ratings.csv')
# Various sizes of data.
num_clusters = len(cluster_df)
num_rows = len(feature_df) # Do not count column names as a row.
# num_keywords = len(str_to_feat(feature_df.at[0,"keyword_vector"]))
# num_genres = len(str_to_feat(feature_df.at[0,"genre_vector"]))
# num_features = num_keywords + num_genres

# Finding the clusters for each book in the rated list.
clusters_to_use = []
clusters_to_use_unique = []
for rated_book in sample_user_books:
    for cluster in range(num_clusters):
        # Flag to check if the book was found.
        found_book_in_cluster = False 
        # Safely evaluate the string representation of the list.
        list_of_books_in_cluster_str = cluster_df.at[cluster, "titles_authors"]
        # Use ast.literal_eval to convert the string to a Python list.
        list_of_books_in_cluster = ast.literal_eval(list_of_books_in_cluster_str)
        for book_in_cluster in list_of_books_in_cluster:
            if rated_book == book_in_cluster:
                clusters_to_use.append(cluster)
                found_book_in_cluster = True
                break # Break from inner loop once book is found in a cluster.
        if found_book_in_cluster:
            break # Break from outer loop once book is found.
# Remove duplicate clusters.
for item in clusters_to_use:
    if item not in clusters_to_use_unique:
        clusters_to_use_unique.append(item)