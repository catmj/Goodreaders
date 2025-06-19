# Goodreaders
Goodreaders project for Erdos Institute Data Science Bootcamp.

Procedural order from data collection to book recommendations:
    Scraping:
        scraping\goodreads_scraping.py
            This script
        scraping\goodreads_scraping_user.py
            This script
        add_reviews_to_books\add_to_books.py
            This script
    Data Splitting:
    Keyword Preprocessing:
    Keyword Processing:
    K-Modes Book Clustering:
        k_means_clustering\k_modes_v2.py
            This script
        k_means_clustering\k_modes_v1.py (older version)
            This script
        k_means_clustering\elbow_method_v2.py
            This script
        k_means_clustering\elbow_method_v1.py (older version)
            This script

Other important scripts:
    large_files.py (various locations)
        Breaks large CSV files into smaller chunks so that GitHub can store them properly.
    recombine_files.py (various locations)
        Recombines file chunks created by "large_files.py" into one CSV file.
    cleaning\create_fake_data.py
        Generates fake feature vectors for a list of title-author pairs (for testing purposes).

    fake_data\genre_list.py
        TO DO
