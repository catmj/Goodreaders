# Goodreaders
Goodreaders project for Erdos Institute Data Science Bootcamp.

Procedural order from data collection to book recommendations:
    1. SCRAPING: Data is collected from "https://www.goodreads.com/" using BeautifulSoup web scraping. Data for individual books is scraped, as well as data for individual user reviews of books.
        scraping\goodreads_scraping.py (scraping\goodreads_scraping.ipynb)

        scraping\goodreads_scraping_user.py (scraping\goodreads_scraping_user.ipynb)

        add_reviews_to_books\add_to_books.py

    2. DATA CLEANING AND MATCHING: Modifies scraped data to make formats more consistent, remove problematic books, etc. User reviews of books are matched with the corresponding books.
        cleaning\same_books_v2.py

    3. DATA SPLITTING: Splits cleaned and matched data into a training dataset and a testing dataset.
        train_test_split\split_data.py

    4. KEYWORD PREPROCESSING (optional): 
        keyword_preprocessing\keyword_preprocessing.py
            Prepares for additional keyword processing by generating a list of potential keywords from each book's description.
    5. KEYWORD PROCESSING: 
        keyword_processing\get_keys_v2.py

    6. KEYWORD CLUSTERING:
        keyword_clustering\cluster_keywords.py

    7. K-MODES BOOK CLUSTERING: 
        k_means_clustering\k_modes_v2.py

        k_means_clustering\elbow_method_v2.py
            Determines optimal number of book clusters for k-modes, using feature vectors of genres and (clustered) keywords for each book. This updated version uses custom dissimilarity metrics with different weights given to keywords and genres.
            Parameters:
                K : int list
                    List of numbers of initial cluster centroids to test with KModes.
                cat_dissim : callable
                    Specifies what function to use as the dissimilarity metric.
                 : CSV file
                    CSV file containing books (titles and authors) and their feature vectors. Two separate feature vectors for genres and (clustered) keywords are generally provided.
                Additional parameters used by KModes.
            Returns:
                "elbow_graph.png" : PNG file
                    The bend in the elbow plot is approximately the optimal number of clusters in terms of costs.
    8A. CONTENT-BASED FILTERING:
        content_filtering\train_similarity_v2.py

    8B. COLLABORATIVE FILTERING:
        collab_filtering\cf_v7.py

    8C. COMBINED FILTERING:
        combined_model\cb_get_recs.py

        combined_model\cf_get_recs.py

        combined_model\combined.py

        combined_model\combined_recs_testing_v3.py

        combined_model\testing_no_print_v2.py

    9. TESTING AND VERIFICATION:


Other important scripts:
    large_files.py (various locations)
        Breaks large CSV files into smaller chunks so that GitHub can store them properly.
    recombine_files.py (various locations)
        Recombines file chunks created by "large_files.py" into one CSV file.
    cleaning\create_fake_data.py
        Generates fake feature vectors for a list of title-author pairs (for testing purposes).

Other scripts not used:
    big_data_set\scripts_older_versions\author_names.py
    big_data_set\scripts_older_versions\combine_v1.py
    cleaning\scripts_older_versions\goodreads_ratings_change_df.py (cleaning\scripts_older_versions\goodreads_ratings_change_df.ipynb)
    cleaning\scripts_older_versions\same_books_v1.py
    collab_filtering\scripts_older_versions\cf_new_user_v1.py
    collab_filtering\scripts_older_versions\cf_new_user_v2.py
    collab_filtering\scripts_older_versions\cf_new_user_v3.py
    collab_filtering\scripts_older_versions\cf_new_user_v4.py
    collab_filtering\scripts_older_versions\cf_new_user_v5.py
    collab_filtering\scripts_older_versions\cf_silly.py
    collab_filtering\scripts_older_versions\cf_v1.py
    collab_filtering\scripts_older_versions\cf_v2.py
    collab_filtering\scripts_older_versions\cf_v3.py
    collab_filtering\scripts_older_versions\cf_v4.py
    collab_filtering\scripts_older_versions\cf_v5.py
    collab_filtering\scripts_older_versions\cf_v6.py
    combined_model\scripts_older_versions\analyze.py
    combined_model\scripts_older_versions\bagging_test.py
    combined_model\scripts_older_versions\combined_recs_testing_v1.py
    combined_model\scripts_older_versions\combined_recs_testing_v2.py
    combined_model\scripts_older_versions\testing_no_print_v1.py
    combined_model\scripts_older_versions\testing.py
    content_filtering\scripts_older_versions\get_recs.py
    content_filtering\scripts_older_versions\train_similarity_v1.py
    k_means_clustering\scripts_older_versions\elbow_method_v1.py
    k_means_clustering\scripts_older_versions\k_modes_v1.py
    keyword_analysis_keyBERT\scripts_older_versions\keyword_extraction_v1.py
    keyword_analysis_keyBERT\scripts_older_versions\keyword_extraction_v2.py
    keyword_processing\scripts_older_versions\get_keys_v1.py

Guide to datasets:







-----------------------------------------------------------------------------------------------------------------

REMAINING TO DO:
    fake_data\genre_list.py
    keyword_analysis_keyBERT\keyword_extraction_v3.py
    collab_filtering\create_dummy_data.py
    big_data_set\combine_v2.py
    big_data_set\find_ids.py
    ANY FILES NOT IN GITHUB

CLEANING PROGRESS:
    add_reviews_to_books
        DONE.
    big_data_set

    cleaned_data
        DONE? Maybe add folders. The difference between this and cleaned_data_2 is unclear.
    cleaned_data_2
        DONE? Maybe add folders. The difference between this and cleaned_data is unclear.
    cleaning

    collab_filtering

    combined_model

    content_filtering

    deliverables
        DONE.
    fake_data

    k_means_clustering
        DONE.
    keyword_analysis_keyBERT

    keyword_clustering

    keyword_preprocessing
        DONE? Data out of date.
    keyword_processing

    scraping

    train_test_split
        DONE? Maybe add folders.
