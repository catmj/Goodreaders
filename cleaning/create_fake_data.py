import pandas as pd # type: ignore
import numpy as np  # type: ignore

books = pd.read_csv('../cleaned_data/books_matched.csv')

books = books[["title", "author_name"]]

book_count = books.shape[0]

fake_keys = np.random.choice(2, (book_count,100),p=[0.9, 0.1])

books["key_clusers"] = fake_keys.tolist()

books.to_csv('../fake_data/books.csv')
