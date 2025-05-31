import pandas as pd # type: ignore
import numpy as np  # type: ignore


books = pd.read_csv('../scraping/books.csv')
reviews = pd.read_csv('../scraping/books_by_user.csv')

books["publication_info"] = pd.to_datetime(books["publication_info"],errors ="coerce")

reviews["date_started"] = pd.to_datetime(reviews["date_started"],errors ="coerce")
reviews["date_added"] = pd.to_datetime(reviews["date_added"],errors ="coerce")
reviews["date_read"] = pd.to_datetime(reviews["date_read"],errors ="coerce")
reviews["date_pub_edition"] = pd.to_datetime(reviews["date_pub_edition"],errors ="coerce")


### cleaning for review author
def clean_author(author_name):
    pieces = author_name.replace("*","").split(", ")
    if len(pieces)==2:
        new_name = pieces[1]+" "+pieces[0]
        return new_name
    else:
        return author_name
    
### not currently accounted for, but in the the books data set, books with two authors are seperated by an and, 
### however the second author name might be cut off by "..."


#cleaning for review title
def clean_title(title):
    first_split = title.split("\n")
    second_split = first_split[0].split(":")
    if second_split[0] == "Black Panther, Vol. 1":
        return "black panther (2016-2018) #1"
    else:
        return second_split[0]
    
#cleaning for book title: observation, cuts off longer titles with "..."
def clean_title_2(title):
    if title == "The Terrible and Wonderful Reasons Why I Run Long Distances (...":
        return "The Terrible and Wonderful Reasons Why I Run Long Distances (Volume 5)"
    elif title == "Death Weavers (4)":
        return "Death Weavers"
    elif title == "The Girl Who Circumnavigated Fairyland in a Ship of Her Own M...":
        return "the girl who circumnavigated fairyland in a ship of her own making"
    elif title == "But What If We're Wrong? Thinking About the Present As If It ...":
       return "but what if we're wrong? thinking about the present as if it were the past"
    elif title == "5 Very Good Reasons to Punch a Dolphin in the Mouth and Other...":
        return "5 very good reasons to punch a dolphin in the mouth and other useful guides"
    else:
        return title



reviews["Author"] = reviews["Author"].map(clean_author).str.lower()
reviews["Title"] = reviews["Title"].map(clean_title).str.lower()
books["author_name"] = books["author_name"].str.lower()
books["title"] = books["title"].map(clean_title_2).str.lower()


pairs_books = set(tuple(x) for x in books[['title', 'author_name']].values)
pairs_reviews = set(tuple(x) for x in reviews[['Title', 'Author']].values)

common_pairs = pairs_books.intersection(pairs_reviews)

books_matched = books[books.apply(lambda row: (row['title'], row['author_name']) in common_pairs, axis=1)]
reviews_matched = reviews[reviews.apply(lambda row: (row['Title'], row['Author']) in common_pairs, axis=1)]
books_unmatched = books[books.apply(lambda row: (row['title'], row['author_name']) not in common_pairs, axis=1)]
reviews_unmatched = reviews[reviews.apply(lambda row: (row['Title'], row['Author']) not in common_pairs, axis=1)]
reviews_unmatched = reviews[["Title","Author"]].drop_duplicates().sort_values(by=['Title'])

books_matched.to_csv('../cleaned_data/books_matched.csv')
reviews_matched.to_csv('../cleaned_data/reviews_matched.csv')
books_unmatched.to_csv('../cleaned_data/books_unmatched.csv')
reviews_unmatched.to_csv('../cleaned_data/reviews_unmatched.csv')
