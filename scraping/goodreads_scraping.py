#This code started from the tutorial at https://dev.to/bekaar_coder/how-to-scrape-data-from-goodreads-using-python-and-beautifulsoup-4mf9

# !pip install beautifulsoup4
# !pip install requests

from bs4 import BeautifulSoup as bs
import requests
import re
import csv

count = 0
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)...",
    "Accept-Language": "en-US, en;q=0.5",
}

#get the list of best books of 2024
app_url = "https://www.goodreads.com"
start_url = "https://www.goodreads.com/choiceawards/best-books-2017" #I think this should work for a lot of years, done 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017

res = requests.get(start_url, headers=HEADERS)
soup = bs(res.text, 'html.parser')

categories = soup.select('.category')
print("number of genres = ", len(categories))

#functions I will need
def clean_string(string):
    cleaned = re.sub(r'\s+', ' ', string).strip()
    return cleaned

def get_ratings_reviews(text):
    # Find the substring for ratings
    ratings = text[:text.find(" ratings")].replace(",", "")

    # Find the substring for reviews
    reviews = text[text.find("and ") + 4:text.find(" reviews")].replace(",", "")

    return int(ratings), int(reviews)

def get_genres(soup): #idea taken from https://stackoverflow.com/questions/75761949/getting-a-book-genres-and-language-scraped-from-goodreads-using-beautifulsoup
    genres = []
    for a in soup.select('a[href*="/genres/"]'):
        genres.append(a.text)
    return genres


#loop through each category of book
for index, category in enumerate(categories):
    genre = category.select('h4.category__copy')[0].text.strip()
    print("genre = ", genre)
    url = category.select('a')[0].get('href')
    category_url = f"{app_url}{url}"

    res = requests.get(category_url, headers=HEADERS)
    soup = bs(res.text, 'html.parser')

    category_books = soup.select('.resultShown a.pollAnswer__bookLink')

#for each book
    for book_index, book in enumerate(category_books):
        parent_tag = book.find_parent(class_='resultShown')
        votes = parent_tag.find(class_='result').text.strip()
        # print(votes)
        book_votes = clean_string(votes).split(" ")[0].replace(",", "")

        #find information about each book
        book_url = book.get('href')
        book_url_formatted = f"{app_url}{book_url}"
        book_img = book.find('img')
        book_img_url = book_img.get('src')
        book_img_alt = book_img.get('alt')
        book_title = clean_string(book_img_alt)
        print(book_title)
        book_name = book_title.split('by')[0].strip()
        book_author = book_title.split('by')[1].strip()

        res = requests.get(book_url_formatted, headers=HEADERS)
        soup = bs(res.text, 'html.parser')

        genres = get_genres(soup)

        #get average book rating
        book_rating = soup.find(class_="RatingStatistics__rating").text.strip()

        #get number of book reviews
        book_ratings_reviews = soup.find(class_="RatingStatistics__meta").get('aria-label').strip()
        book_ratings, book_reviews = get_ratings_reviews(book_ratings_reviews)
        print(f"Ratings: {book_ratings}, Reviews: {book_reviews}")

        #get book blurb
        book_description_elements = soup.select('.BookPageMetadataSection__description .Formatted')
        if book_description_elements:
            book_description = book_description_elements[0].text
        else:
            book_description = 'No description found'

        #find number of people wanting to and currently reading
        text = soup.get_text()

        # Use regex to find the number before "people want to read"
        match = re.search(r'([\d,]+)\s+people want to read', text)

        if match:
            number_str = match.group(1).replace(",", "")
            want_to_read_num = int(number_str)
        else:
            want_to_read_num = -1

        # Use regex to find the number before "people are currently reading"
        match = re.search(r'([\d,]+)\s+people are currently reading', text)

        if match:
            number_str = match.group(1).replace(",", "")
            current_read_num = int(number_str)
        else:
            current_read_num = -1

        #add this histogram of book ratings
        book_ratings_reviews_hist = soup.find_all(class_="RatingsHistogram__labelTotal")#.get('aria-label').strip()
        hist = []
        for i in book_ratings_reviews_hist:
            hist.append(i.text.strip())

        #add the first page of book reviews
        book_ratings_reviews_written = soup.find_all(class_="TruncatedContent__text TruncatedContent__text--large")#.get('aria-label').strip()
        written_reviews = []
        for i in book_ratings_reviews_written:
            written_reviews.append(i.text.strip())

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

        book_dict = {
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
            "book_url": f"{app_url}{book_url}",
            "ratings_hist": hist,
            # "written_reviews": written_reviews, #written reviews is giveing me problems right now with the csv
            "want_to_read_num": want_to_read_num,
            "current_read_num": current_read_num,
        }

        # for i in book_dict:
        #     print(i, ":", book_dict[i])

        #     print(book_dict.keys())

        csv_filename = "books.csv"

        #get titles when starting new file
        # if index == 0 and book_index == 0:
        #     with open(csv_filename, "w", newline="", encoding="utf-8") as csv_file:
        #         writer = csv.DictWriter(csv_file, fieldnames=book_dict.keys(), quoting=csv.QUOTE_ALL)
        #         writer.writeheader()


        with open(csv_filename, mode="a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=book_dict.keys(), quoting=csv.QUOTE_ALL)
            writer.writerow(book_dict)


