# Goodreaders
Goodreaders project for Erdos Institute Data Science Bootcamp.

Procedural order from data collection to book recommendations and verification:
    1. SCRAPING: Collect data from "https://www.goodreads.com/" using BeautifulSoup web scraping. Scrape data for individual books, as well as data for individual user reviews of books.
        scraping\goodreads_scraping.py (scraping\goodreads_scraping.ipynb)
            Scrapes book data from Goodreads' yearly Readers' Favorite Books lists at "https://www.goodreads.com/choiceawards/best-books-XXXX", where XXXX is a year between 2011-2024. Will likely work for future years.
            Parameters:
                start_url : str (URL)
                    Readers' Favorite Books end-of-year list webpage to scrape.
            Returns:
                "books.csv" : CSV file
                    Contains data for each book scraped, such as title, author, description, average rating, and genres.
        scraping\goodreads_scraping_user.py (scraping\goodreads_scraping_user.ipynb)
            Scrapes user book review data from the Goodreads pages of randomly selected Goodreads users.
            Parameters:
                base_url : str (URL)
                    The webpage from which to scrape a user's ratings and reviews. The exact URL differs by user ID.
            Returns:
                "books_by_user.csv" : CSV file
                    Contains data for each user review scraped, such as book title, book author, user ID, user rating, average rating, and review text.
        add_reviews_to_books\add_to_books.py
            Scrapes book data for books that do not appear on Goodreads' Readers' Favorite Books lists but do have ratings scraped by "goodreads_scraping_user.py".
            Parameters:
                "reviews_with_urls_left_to_scrape.csv" : str (CSV file)
                    Each row corresponds to a book and contains the portion of its URL after "https://www.goodreads.com/book/show/". Some books have duplicate URLs, particularly books whose titles contain non-Latin characters.
            Returns:
                "books_in_progress.csv" : CSV file
                    Adds a row of data for each book scraped, one at a time. Can add rows to an existing file "books_in_progress.csv".
    2. DATA CLEANING AND MATCHING: Modify scraped data to make formats more consistent, remove problematic books, etc. Match user reviews of books with the corresponding books.
        cleaning\same_books_v2.py

            Parameters:
        
            Returns:
    
    3. DATA SPLITTING: Split cleaned and matched data into a training dataset and a testing dataset. Use the latter for verification purposes.
        train_test_split\split_data.py

            Parameters:

            Returns:

    4. KEYWORD PREPROCESSING (optional): Create a list of potential keywords from book descriptions by removing common and problematic words. Reformat book descriptions to ensure consistency.
        keyword_preprocessing\keyword_preprocessing.py
            Takes words from books' descriptions and users' reviews, formats them consistently, removes special characters and numerals, and removes short words and words commonly found in book reviews (stopwords).
            Parameters:
                "books_matched_copy.csv" : str (CSV file)
                    Contains an indexed list of books, where each book has a description ("description").
                "reviews_matched_copy.csv" : str (CSV file)
                    Contains an indexed list of user reviews of books, where each user review may have user-written text ("review").
                new_stop_words : str list
                    Additional stopwords to add to the default list of stopwords from the nltk.corpus library. Generally, contains words common in book descriptions and reviews, such as words related to reading, publishing, or writing.
            Returns: 
                "output_file_books.csv" : CSV file
                    Contains all potential keywords from each book description from the "books_matched_copy.csv" dataset.
                "output_file_reviews.csv" : CSV file
                    Contains all potential keywords from each user review from the "reviews_matched_copy.csv" dataset. Not as useful since book reviews are usually blank (ratings only).
    5. KEYWORD PROCESSING: Create a list of keywords from each book's description.
        keyword_processing\get_keys_v2.py

            Parameters:

            Returns:

    6. KEYWORD CLUSTERING: Create clusters of keywords that have similar meanings. Assign each cluster a component in the feature vectors.
        keyword_clustering\cluster_keywords.py

            Parameters:

            Returns:

    7. K-MODES BOOK CLUSTERING (optional): Create clusters of similar books using their feature vectors and the k-modes algorithm (a variant of k-means for categorical data). Use these clusters to reduce runtime when making book recommendations.
        k_means_clustering\k_modes_v2.py
            Generates a fixed number of book clusters using k-modes, using feature vectors of genres and (clustered) keywords for each book. This updated version uses custom dissimilarity metrics with different weights given to keywords and genres.
            Parameters:
                num_clusters : int
                    The predetermined number of clusters to form using KModes.
                cat_dissim : callable
                    Specifies the function to use as the dissimilarity metric (e.g., hamming_dist, cosine_dissim).
                "books_with_vectors_and_sums_and_ratings.csv" : str (CSV file)
                    CSV file containing books (titles and authors) and their feature vectors. Two separate feature vectors for genres and (clustered) keywords are generally provided. Also includes number of ratings and average rating for each book.
                keyword_weight : float
                    How much keywords should be weighted relative to genres when measuring similarities. The optimal is somewhere around 0.8.
                Additional parameters used by KModes (e.g., init, n_init, max_iter, verbose).
            Returns:
                "books_clustered.csv" : CSV file
                    The input CSV file with an additional column indicating the assigned cluster for each book.
                "books_by_cluster.csv" : CSV file
                    CSV file containing lists of books, grouped by their assigned cluster.
        k_means_clustering\elbow_method_v2.py
            Determines optimal number of book clusters for k-modes, using feature vectors of genres and (clustered) keywords for each book. This updated version uses custom dissimilarity metrics with different weights given to keywords and genres.
            Parameters:
                K : int list
                    List of numbers of initial cluster centroids to test with KModes.
                cat_dissim : callable
                    Specifies the function to use as the dissimilarity metric (e.g., hamming_dist, cosine_dissim).
                "books_with_vectors_and_sums.csv" : str (CSV file)
                    CSV file containing books (titles and authors) and their feature vectors. Two separate feature vectors for genres and (clustered) keywords are generally provided.
                keyword_weight : float
                    How much keywords should be weighted relative to genres when measuring similarities. The optimal is somewhere around 0.8.
                Additional parameters used by KModes (e.g., init, n_init, max_iter, verbose).
            Returns:
                "elbow_graph.png" : PNG file
                    The bend in the elbow plot is approximately the optimal number of book clusters in terms of costs.
        k_means_clustering\small_read_list_recommendations.py
            Given a list of books that a user has read and enjoyed (no ratings are required to be specified), generates a list of popular books that share a k-modes cluster with any book in the list. Designed to be a simpler recommendation system for readers who want to provide a shorter list of books they have read. 
            Parameters:
                sample_user_books : str list
                    An unrated list of books (titles and authors) that a user has read before.
                "books_clustered_ratings.csv" : str (CSV file)
                    CSV file containing books (titles and authors) and their feature vectors, with an additional column indicating the assigned cluster for each book. Also includes number of ratings and average rating for each book.
                "books_by_cluster_ratings.csv" : str (CSV file)
                    CSV file containing lists of books, grouped by their assigned cluster. Must correspond to the clusters in "books_clustered_ratings.csv".
                std_num_outlier : float
                    The number of standard deviations below the mean similarity that serves as the threshold for outliers.
                highest_hits_to_keep : int
                    The number of books with the highest number of ratings to keep. Should be very high relative to the number of books left after filtering for outliers.
                highest_ratings_to_keep : int
                    The number of the most highly rated books (in terms of average rating) to keep. Should be less than "highest_hits_to_keep".
                keyword_weight : float
                    How much keywords should be weighted relative to genres when measuring similarities. The optimal is somewhere around 0.8.
            Returns:
                "recommendations.csv" : CSV file
                    Popular books that share a k-modes cluster with a book the user has read before, sorted by average rating.
    8A. CONTENT-BASED FILTERING: Generate book recommendation lists using content-based filtering (option A).
        content_filtering\train_similarity_v2.py

            Parameters:

            Returns:

    8B. COLLABORATIVE FILTERING: Generate book recommendation lists using collaborative filtering (option B).
        collab_filtering\cf_v7.py

            Parameters:

            Returns:

    8C. COMBINED FILTERING: Generate book recommendation lists using a combination of content-based filtering and collaborative filtering (option C).
        combined_model\cb_get_recs.py

            Parameters:

            Returns:

        combined_model\cf_get_recs.py

            Parameters:

            Returns:

        combined_model\combined.py

            Parameters:

            Returns:

        combined_model\combined_recs_testing_v3.py

            Parameters:

            Returns:

        combined_model\testing_no_print_v2.py

            Parameters:

            Returns:

    9. TESTING AND VALIDATION: Verify the accuracy of book recommendation lists.
        MANUAL TESTING
            The quality of book recommendation lists can be verified by developers by inputting a list of books with ratings out of five stars and then evaluating the quality of the resulting recommendations.

Other important scripts:
    large_files.py (various locations)
        Breaks large CSV files into smaller chunks so that GitHub can store them properly.
    recombine_files.py (various locations)
        Recombines file chunks created by "large_files.py" into one CSV file.
    cleaning\create_fake_data.py
        Generates fake feature vectors for a list of title-author pairs (for testing purposes).
    app\app.py
        Creates an app with a user interface that makes the recommendation process easier.
    app\app_v2.py
        Creates an app with a user interface that makes the recommendation process easier. Includes a k-modes recommender.

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
    1. SCRAPING:
        scraping\books_by_user.csv
            Contains data for each user review scraped, such as book title, book author, user ID, user rating, average rating, and review text.
        scraping\books.csv
            Contains data for each book scraped from the Goodreads' yearly Readers' Favorite Books lists, such as title, author, description, average rating, and genres.
        add_reviews_to_books\working_folder\books_in_progress.csv
            Contains data for each book scraped, including those not on the Goodreads' yearly Readers' Favorite Books lists, such as title, author, description, average rating, and genres. Some information from the Goodreads' Readers' Favorite Books lists is not present for all books, such as number of votes and award category.
        add_reviews_to_books\working_folder\reviews_with_urls_left_to_scrape.csv
            URLs of books to be scraped. In particular, those books that do have user reviews but are not present on the Goodreads' Readers' Favorite Books lists.
        add_reviews_to_books\working_folder\reviews_with_urls_not_included.csv
            URLs of books that were not scraped due to causing issues. Non-Latin text was the most common cause of issues.
        add_reviews_to_books\duplicate_url_identification\reviews_unmatched_with_urls_filtered_no_dupes.csv
            URLs for all books that do not have problematic duplicate URLs.
        add_reviews_to_books\duplicate_url_identification\reviews_unmatched_with_urls_filtered_only_dupes.csv
            URLs for all books that do have problematic duplicate URLs.
        add_reviews_to_books\final_output_books_unmatched_uncleaned\books_full_unmatched_uncleaned.csv

    2. DATA CLEANING AND MATCHING:

    3. DATA SPLITTING:
    
    4. KEYWORD PREPROCESSING (optional):

    5. KEYWORD PROCESSING:

    6. KEYWORD CLUSTERING:

    7. K-MODES BOOK CLUSTERING:

    8A. CONTENT-BASED FILTERING:

    8B. COLLABORATIVE FILTERING:

    8C. COMBINED FILTERING:

    9. TESTING AND VALIDATION:












-----------------------------------------------------------------------------------------------------------------

REMAINING TO DO:
    fake_data\genre_list.py
    keyword_analysis_keyBERT\keyword_extraction_v3.py
    collab_filtering\create_dummy_data.py
    big_data_set\combine_v2.py
    big_data_set\find_ids.py
    ANY FILES NOT IN GITHUB
    VERSIONS?

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
