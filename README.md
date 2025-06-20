# Goodreaders
Goodreaders project for Erdos Institute Data Science Bootcamp.

Procedural order from data collection to book recommendations:
    1. SCRAPING: 
        scraping\goodreads_scraping.py (scraping\goodreads_scraping.ipynb)

        scraping\goodreads_scraping_user.py (scraping\goodreads_scraping_user.ipynb)

        add_reviews_to_books\add_to_books.py

    2. DATA CLEANING AND MATCHING: Modifies scraped data to make formats more consistent, remove problematic books, etc.

    3. DATA SPLITTING: 
        train_test_split\split_data.py

    4. KEYWORD PREPROCESSING (optional): 
        keyword_preprocessing\keyword_preprocessing.py
            Prepares for additional keyword processing by generating a list of potential keywords for each book.
    5. KEYWORD PROCESSING: 

    6. KEYWORD CLUSTERING:
        keyword_clustering\cluster_keywords.py

    7. K-MODES BOOK CLUSTERING: 
        k_means_clustering\k_modes_v2.py

        k_means_clustering\k_modes_v1.py (older version)

        k_means_clustering\elbow_method_v2.py

        k_means_clustering\elbow_method_v1.py (older version)

    8A. CONTENT-BASED FILTERING:

    8B. COLLABORATIVE FILTERING:

    8C. COMBINED FILTERING:

    9. VERIFICATION:


Other important scripts:
    large_files.py (various locations)
        Breaks large CSV files into smaller chunks so that GitHub can store them properly.
    recombine_files.py (various locations)
        Recombines file chunks created by "large_files.py" into one CSV file.
    cleaning\create_fake_data.py
        Generates fake feature vectors for a list of title-author pairs (for testing purposes).








REMAINING TO DO:
    fake_data\genre_list.py
    keyword_processing\get_keys.py
    keyword_processing\get_keys_2.py
    keyword_analysis_keyBERT\keyword_extraction.py
    keyword_analysis_keyBERT\keyword_extraction_v2.py
    keyword_analysis_keyBERT\key_word_extraction_v3.py
    content_filtering\get_recs.py
    content_filtering\train_similarity.py
    content_filtering\train_similarity_2.py
    collab_filtering\cf_new_user.py
    collab_filtering\cf_new_user_v2.py
    collab_filtering\cf_new_user_v3.py
    collab_filtering\cf_new_user_v4.py
    collab_filtering\cf_new_user_v5.py
    collab_filtering\cf_silly.py
    collab_filtering\cf_v1.py
    collab_filtering\cf_v2.py
    collab_filtering\cf_v3.py
    collab_filtering\cf_v4.py
    collab_filtering\cf_v5.py
    collab_filtering\cf_v6.py
    collab_filtering\cf_v7.py
    collab_filtering\create_dummy_data.py
    combined_model\analyze.py
    combined_model\bagging_test.py
    combined_model\cb_get_recs.py
    combined_model\cf_get_recs.py
    combined_model\combined_recs_testing.py
    combined_model\combined_recs_testing_2.py
    combined_model\combined_recs_testing_3.py
    combined_model\combined.py
    combined_model\testing_no_print.py
    combined_model\testing_no_print_2.py
    combined_model\testing.py
    cleaning\same_books.py
    cleaning\same_books_v2.py
    cleaning\goodreads_ratings_change_df.py (cleaning\goodreads_ratings_change_df.ipynb)
    big_data_set\author_names.py
    big_data_set\combine.py
    big_data_set\combine2.py
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
