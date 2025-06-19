# Goodreaders
Goodreaders project for Erdos Institute Data Science Bootcamp.

Procedural order from data collection to book recommendations:
    1. SCRAPING: 
        scraping\goodreads_scraping.py

        scraping\goodreads_scraping_user.py

        add_reviews_to_books\add_to_books.py

    2. DATA CLEANING: Modifies scraped data to make formats more consistent, remove problematic books, etc.

    3. DATA SPLITTING: 

    4. KEYWORD PREPROCESSING: 
        keyword_preprocessing\keyword_preprocessing.py
            Prepares for additional keyword processing by generating a list of potential keywords for each book.
    5. KEYWORD PROCESSING: 

    6. K-MODES BOOK CLUSTERING: 
        k_means_clustering\k_modes_v2.py

        k_means_clustering\k_modes_v1.py (older version)

        k_means_clustering\elbow_method_v2.py

        k_means_clustering\elbow_method_v1.py (older version)


Other important scripts:
    large_files.py (various locations)
        Breaks large CSV files into smaller chunks so that GitHub can store them properly.
    recombine_files.py (various locations)
        Recombines file chunks created by "large_files.py" into one CSV file.
    cleaning\create_fake_data.py
        Generates fake feature vectors for a list of title-author pairs (for testing purposes).

REMAINING TO DO:
    fake_data\genre_list.py
        TO DO
