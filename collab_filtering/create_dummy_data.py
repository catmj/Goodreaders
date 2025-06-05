import pandas as pd # type: ignore
import numpy as np  # type: ignore

books = pd.read_csv('../cleaned_data/books_matched.csv')

books = books[["title", "author_name"]].reset_index(drop=True)

book_count = books.shape[0]

fake_keys = np.random.choice(2, (book_count,100),p=[0.9, 0.1])
fake_average_rating = np.round(np.random.uniform(low=1, high = 5, size = book_count),2)

books["key_clusers"] = fake_keys.tolist()
books["rating"] = fake_average_rating.tolist()


books.to_csv('../fake_data/books.csv', index=False)


user_count = book_count*10

users = []

for i in range(user_count):
    user_id = np.random.randint(100000)
    num_books = np.random.randint(3,15)


    user = books[["title", "author_name"]].sample(num_books)
    user["Rating"] = np.random.randint(1,5,num_books)
    user["User_id"] = np.ones(num_books)*user_id

    users.append(user)

user_frame = pd.concat(users)

user_frame = user_frame.rename(columns={"title":"Title","author_name":"Author"})
user_frame = user_frame.reset_index(drop=True)

user_frame.to_csv("../fake_data/ratings.csv", index=False)

