# Prepares for additional keyword processing by generating a list of potential keywords for each book.
# Good resource: https://www.kaggle.com/code/akhatova/extract-keywords

# To run in a virtual environment, first do: pip install -r requirements.txt
# PACKAGES
import numpy as np
import pandas as pd
import re
# May need to run these three lines the first time:
# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# IMPORTING COPIES OF DATA
df1 = pd.read_csv('./keyword_analysis/books_matched_copy.csv')
df2 = pd.read_csv('./keyword_analysis/reviews_matched_copy.csv')
# df1.info()
# df2.info()

# PRE-PROCESSING
stop_words = set(stopwords.words('english'))
# Choose stopwords to exclude.
new_stop_words = ["book", "books", "novel", "novels", "writer", "writers", "write", "writes", "writing", "written", "author", "authors", "edition", "editions",
                  "bestseller", "bestsellers", "bestselling", "reader", "readers", "reading", "read", "reads", "isbn", "title", "titles", "titled", "titular",   
                  "genre", "genres", "review", "reviews", "reviewed", "reviewer", "reviewers", "critic", "critics", "series", "newest", "latest", "literary", 
                  "hardcover", "paperback", "publisher", "publishers", "published", "yet"]
# Add the above words to the list of default stopwords.
stop_words = list(stop_words.union(new_stop_words))
def pre_process(text):
    # Make text lowercase.
    text = text.lower()
    # Remove special characters and digits.
    text = re.sub("(\\d|\\W)+"," ",text)
    # May also need to remove tags?
    text = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    # Convert text from string to list.
    text = text.split()
    # Remove stopwords from text.
    text = [word for word in text if word not in stop_words]
    # Remove short words (less than 3 letters).
    text = [word for word in text if len(word) >= 3]
    # Lemmatize.
    lmtzr = WordNetLemmatizer()
    text = [lmtzr.lemmatize(word) for word in text]
    return ' '.join(text)

# KEYWORDS FOR BOOKS.CSV
keyword_lists_df1 = df1['description'].astype(str).apply(lambda x:pre_process(x)) # Create keyword lists.
# print(keyword_lists_df1)
keyword_lists_df1.to_csv('./keyword_analysis/output_file_books.csv', index=True) # Convert keyword lists to .csv file.

# KEYWORDS FOR REVIEWS.CSV
keyword_lists_df2 = df2['review'].astype(str).apply(lambda x:pre_process(x)) # Create keyword lists.
# print(keyword_lists_df2)
keyword_lists_df2.to_csv('./keyword_analysis/output_file_reviews.csv', index=True) # Convert keyword lists to .csv file.

# OPTIONAL TESTING CODE
# mystr = "This is my books test line. Mathematics. Physics. Giraffe. Pineapple. Helpers."
# newstr = pre_process(mystr)
# print(newstr)
# print(type(newstr))