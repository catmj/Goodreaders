import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


#### None of this needs to be done again in real version
ratings = pd.read_csv('../fake_data/ratings.csv')

ratings["book"] = ratings[['Title', 'Author']].agg(', '.join, axis=1)

ratings = ratings[["book","User_id","Rating"]]

ratings = ratings[ratings['Rating'] != 'Invalid rating']
ratings['Rating'] = ratings['Rating'].astype(int)


pivot_ratings = ratings.pivot_table(index='book', columns='User_id', values='Rating')

Y = pivot_ratings.to_numpy()


num_books, num_users = Y.shape
num_features = 200

Ymean = np.nanmean(Y,axis=1)
Ynorm = (Y.T-Ymean).T

X = np.random.rand(num_books, num_features) #probably should be a csv file

book_list = pivot_ratings.index.tolist() #ditto


####Everything above here should be from training


my_books = ["gideon the ninth, tamsyn muir", "the fifth season, n.k. jemisin", 
            "return of the thief, megan whalen turner", "iron widow, xiran jay zhao",
            "catching fire, suzanne collins"]

my_ratings = [5,5,4,3,3]
my_index = [book_list.index(book) for book in my_books]
my_index_rating = list(zip(my_index,my_ratings))

test_ratings = np.full(num_books, np.nan)

for i,rating in my_index_rating:
    test_ratings[i] = rating


test_rated = [i for i in range(len(test_ratings)) if test_ratings[i] > 0]

print('\nTest user ratings:\n')
for i in range(len(test_ratings)):
    if test_ratings[i] > 0 :
        print(f'Rated {test_ratings[i]} for  {book_list[i]}')

y = test_ratings-Ymean

m = ~(np.isnan(y))  
X_m, y_m = X[m], y[m]                         
reg = LinearRegression().fit(X_m, y_m)

W = reg.coef_
b = reg.intercept_

my_predictions = np.matmul(X,W)+b
my_predictions = my_predictions+Ymean

# sort predictions
evals = list(zip(range(num_books),my_predictions))
ix = sorted(evals, key=lambda x: x[1], reverse=True)

for i in range(17):
    j = ix[i][0]
    if j not in test_rated:
        print(f'Predicting rating {my_predictions[j]:0.2f} for book {book_list[j]}')

print('\n\nOriginal vs Predicted ratings:\n')
for i in range(len(test_ratings)):
    if test_ratings[i] > 0:
        print(f'Original {test_ratings[i]}, Predicted {my_predictions[i]:0.2f} for {book_list[i]}')