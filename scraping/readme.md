5/20/25

The jupyter notebook was me playing around to make it work, and is probably a little messy
The .py actually generates a dataset. It is building off of this tutorial: https://dev.to/bekaar_coder/how-to-scrape-data-from-goodreads-using-python-and-beautifulsoup-4mf9
It reads the books from best-books-2024 and makes a .csv of all the info after scrapping each book's webpage. Could add a lot more books to this dataset, but this is a good start. 

as of this push, has all books on best-books from 2019 - 2024. 2511 books

The files with _user on them are pulled from user pages and each line is one book with one user (given by id). Seems like the way to do reviews, indvidual ratings, editions, etc. 
Have a random number generator to get users, but it will take a while and be a lot of data. 

idk how cleaned up these datasets are right now. 