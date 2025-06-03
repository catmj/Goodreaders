# Data after entry 5336 may still need to be cleaned or matched.
# TO FIX: URLs with multiple books.

# Needed packages.
from bs4 import BeautifulSoup as bs
import requests
import re
import csv
import pandas as pd

# Input data.
count = 0
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)...",
    "Accept-Language": "en-US, en;q=0.5",
}
url_base = "https://www.goodreads.com/book/show/"
df = pd.read_csv('reviews_unmatched_with_goodreads_urls_filtered_no_dupes.csv')

# Function for cleaning strings.
def clean_string(string):
    cleaned = re.sub(r'\s+', ' ', string).strip()
    return cleaned

# Function for getting reviews and ratings.
def get_ratings_reviews(text): 
    # Find the substring for ratings.
    ratings = text[:text.find(" ratings")].replace(",", "")
    # Remove non-numeric characters.
    ratings = re.sub("[^0-9]", "", ratings)
    # Find the substring for reviews.
    reviews = text[text.find("and ") + 4:text.find(" reviews")].replace(",", "")
    # Remove non-numeric characters.
    reviews = re.sub("[^0-9]", "", reviews)
    return int(ratings), int(reviews)

# Function for getting genres.
def get_genres(soup): # Idea taken from https://stackoverflow.com/questions/75761949/getting-a-book-genres-and-language-scraped-from-goodreads-using-beautifulsoup.
    genres = []
    for a in soup.select('a[href*="/genres/"]'):
        genres.append(a.text)
    return genres

# Scraping each book's page.
for index, row in df.iterrows():
    # if index < 8: # Testing only.
    url_addendum = df.astype(str).iloc[index,3]
    start_url = url_base + url_addendum 
    # book_url_formatted = f"{start_url}"
    # print(start_url)

    res = requests.get(start_url, headers=HEADERS)
    soup = bs(res.text, 'html.parser')
    # print(soup)

    # Missing categories (included in original books.csv dataset).
    book_votes = "Not applicable"
    genre = "Not applicable"
    book_img_url = "Missing"

    # Get easy information.
    genres = get_genres(soup)
    book_name = df.astype(str).iloc[index,1]
    book_author = df.astype(str).iloc[index,2]
    book_rating = soup.find(class_="RatingStatistics__rating").text.strip()

    # Get number of book reviews.
    book_ratings_reviews = soup.find(class_="RatingStatistics__meta").get('aria-label').strip()
    book_ratings, book_reviews = get_ratings_reviews(book_ratings_reviews)
    # print(f"Ratings: {book_ratings}, Reviews: {book_reviews}")

    # Get book blurb.
    book_description_elements = soup.select('.BookPageMetadataSection__description .Formatted')
    if book_description_elements:
        book_description = book_description_elements[0].text
    else:
        book_description = 'No description found'

    # Find number of people wanting to and currently reading.
    text = soup.get_text()

    # Use regex to find the number before "people want to read".
    match = re.search(r'([\d,]+)\s+people want to read', text)

    if match:
        number_str = match.group(1).replace(",", "")
        want_to_read_num = int(number_str)
    else:
        want_to_read_num = -1

    # Use regex to find the number before "people are currently reading".
    match = re.search(r'([\d,]+)\s+people are currently reading', text)
    if match:
        number_str = match.group(1).replace(",", "")
        current_read_num = int(number_str)
    else:
        current_read_num = -1

    # Add histogram of book ratings.
    book_ratings_reviews_hist = soup.find_all(class_="RatingsHistogram__labelTotal")#.get('aria-label').strip()
    hist = []
    for i in book_ratings_reviews_hist:
        hist.append(i.text.strip())

    # Other information.
    author_avatar_url_element = soup.select('.PageSection .AuthorPreview a.Avatar img.Avatar__image')
    if author_avatar_url_element:
        author_avatar_url = author_avatar_url_element[0].get('src')
    else:
        author_avatar_url = 'No Avatar found'
    author_description_element = soup.select('.PageSection > .TruncatedContent .Formatted')
    if author_description_element:
        author_description = author_description_element[0].text
    else:
        author_description = 'No description found'
    bookPages = soup.select_one(".FeaturedDetails p[data-testid='pagesFormat']")
    if bookPages:
        pages_format = bookPages.text[:bookPages.text.find(" pages")]
    else:
        pages_format = "No pages found"
    publication_info = soup.select_one(".FeaturedDetails p[data-testid='publicationInfo']")
    if publication_info:
        publication = publication_info.text[16:]
    else:
        publication = "No publication found"

    index_to_use = index + 5337
    book_dict = {
        "index": index_to_use,
        "category": genre,
        "genres": genres,
        "votes": book_votes,
        "title": book_name,
        "description": book_description, 
        "author_name": book_author,
        "author_about": author_description, 
        "avatar_url": author_avatar_url, 
        "pages": pages_format, 
        "rating": book_rating, 
        "num_ratings": book_ratings, 
        "num_reviews": book_reviews, 
        "publication_info": publication, 
        "img_url": book_img_url, 
        "book_url": start_url,
        "ratings_hist": hist, 
        "want_to_read_num": want_to_read_num, 
        "current_read_num": current_read_num, 
        }

    # for i in book_dict:
    #     print(i, ":", book_dict[i])
    # print(book_dict.keys())

    csv_filename = "books_matched_more.csv"
    with open(csv_filename, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=book_dict.keys(), quoting=csv.QUOTE_ALL)
        writer.writerow(book_dict)