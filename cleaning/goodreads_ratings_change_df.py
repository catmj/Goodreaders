import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import numpy as np
import random
import pickle 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import csv

df = pd.read_csv('../cleaned_data/reviews_matched.csv')

#print out some info about the data
print(df.shape)
df = df.drop_duplicates()
print(df.shape)
# Print the entire DataFrame
print(df)

duplicate_count = df['Title'].duplicated().sum()
# print("Number of books with at least two people rating it:", duplicate_count)
duplicate_counts_per_value = df['Title'].value_counts()
print(duplicate_counts_per_value)
titles = list(df['Title'].unique())
num_titles = len(titles)
print("Number of unique books: ", num_titles)
user_ids = list(df['User_id'].unique())
num_users = len(user_ids)
print("number of users is: ", num_users)
# print("user_ids = ", user_ids)
#Number of reviews per title
print("Num reviews, title")
for index, value in duplicate_counts_per_value.iteritems():
        print(value, index)

# Convert 'rating' to numeric, forcing invalid values to NaN
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

# Remove rows where 'rating' is NaN
df = df.dropna(subset=["Rating"]).reset_index(drop=True)

# Group by title to find count and average rating
result = df.groupby("Title").agg(
    count=("Title", "size"),
    avg_rating=("Rating", "mean")
).reset_index()

# Make an array with columns as users and rows as books
ratings = np.zeros((num_users, num_titles))

for index, row in df.iterrows():
    if row['Title'] in titles:
        try:
            ratings[user_ids.index(row['User_id']), titles.index(row["Title"])] = int(row["Rating"])
        except:
            pass
        
print("ratings size = ", ratings.shape)
#remove users where all lines are 0s. 
ratings = ratings[~np.all(ratings == 0, axis=1)]
print("ratings size = ", ratings.shape)

with open("../cleaned_data/ratings_user_v_book.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(ratings)

# Create a custom colormap with white, yellow, and red
cmap = mcolors.ListedColormap(['white', 'red'])
bounds = [0, 0.1, 5]  # Define ranges: 0 (white), 0.1-0.5 (yellow), 0.5-1 (red)
norm = mcolors.BoundaryNorm(bounds, len(cmap.colors))

plt.imshow(ratings, cmap=cmap, norm=norm)
plt.colorbar()
plt.xlabel("Books")
plt.ylabel("Readers")
plt.show()

plt.hist(duplicate_counts_per_value/num_users*100, bins = 32)
plt.ylim([0,50])
plt.xlabel('% of readers who read each book')
plt.ylabel("Number of books")
# print(duplicate_counts_per_value)

duplicate_counts_per_user = df['User_id'].value_counts()
plt.hist(duplicate_counts_per_user, bins = 100)
plt.ylim([0,75])
plt.xlim([0,1000])
plt.xlabel('Number of reads per user')
plt.ylabel("Number of users")
# print(duplicate_counts_per_value)