# Goodreaders

[](https://github.com/catmj/Goodreaders#goodreaders)

Goodreaders book recommender project for The Erdos Institute Data Science Bootcamp. By Catherine Johnston, Yevgeniya “Jonah” Tarasova, Katie Johnston, and Ava Bamforth.


Procedural order from data collection to book recommendations and verification: 


**1. SCRAPING:** 
Collect data from `https://www.goodreads.com/` using BeautifulSoup web scraping. Scrape data for individual books, as well as data for individual user reviews of books.

-   `scraping\goodreads_scraping.py` (`scraping\goodreads_scraping.ipynb`)

-   Scrapes book data from Goodreads' yearly Readers' Favorite Books lists at `https://www.goodreads.com/choiceawards/best-books-XXXX`, where `XXXX` is a year between 2011-2024. Will likely work for future years.

-   **Parameters:**

    -   `start_url`: str (URL)

        -   Readers' Favorite Books end-of-year list webpage to scrape.

-   **Returns:**

    -   `books.csv`: CSV file

        -   Contains data for each book scraped, such as title, author, description, average rating, and genres.

-   `scraping\goodreads_scraping_user.py` (`scraping\goodreads_scraping_user.ipynb`)

-   Scrapes user book review data from the Goodreads pages of randomly selected Goodreads users.

-   **Parameters:**

    -   `base_url`: str (URL)

        -   The webpage from which to scrape a user's ratings and reviews. The exact URL differs by user ID.

-   **Returns:**

    -   `books_by_user.csv`: CSV file

        -   Contains data for each user review scraped, such as book title, book author, user ID, user rating, average rating, and review text.

-   `add_reviews_to_books\add_to_books.py`

-   Scrapes book data for books that do not appear on Goodreads' Readers' Favorite Books lists but do have ratings scraped by `goodreads_scraping_user.py`.

-   **Parameters:**

    -   `reviews_with_urls_left_to_scrape.csv`: str (CSV file)

        -   Each row corresponds to a book and contains the portion of its URL that comes after `https://www.goodreads.com/book/show/`. Some books have duplicate URLs, particularly books whose titles contain non-Latin characters.

-   **Returns:**

    -   `books_in_progress.csv`: CSV file

        -   Adds a row of data for each book scraped, one at a time. Can add rows to an existing file `books_in_progress.csv`.


**2. DATA CLEANING AND MATCHING:** 
Modify scraped data to make formats more consistent, remove problematic books, etc. Match user reviews of books with the corresponding books. 

-   `cleaning\same_books_v2.py`

-   Cleans datasets for books and user reviews, and then matches each review to its corresponding book.

- **Parameters:**

    -   `books_full_unmatched_uncleaned.csv`: str (CSV file)

        -   Uncleaned and unmatched books dataset.

    -   `books_by_user.csv`: str (CSV file)

        -   Uncleaned and unmatched user reviews dataset.

- **Returns:**

    -   `books_matched`: CSV file

        -   Cleaned and matched books dataset.

    -   `reviews_matched`: CSV file
    
        -   Cleaned and matched user reviews dataset.


**3. DATA SPLITTING:**
Split cleaned and matched data into a training dataset and a testing dataset. Use the latter for verification purposes. 

-   `train_test_split\split_data.py`
    
-   Splits cleaned and matched data into training, testing, and cross-validation datasets. It performs user-based splitting for reviews, ensuring testable users have sufficient reviews, and matches book entries across splits for consistency. The output files are also chunked for easier handling.
    
-   **Parameters:**
    
    -   `DATA_PATH`: str (directory path)
        
        -   Path to the directory containing the cleaned and matched input CSV files.
            
    -   `RATINGS_FILE`: str (CSV file path)
        
        -   Path to the `reviews_matched.csv` file within `DATA_PATH`.
            
    -   `BOOK_FILE`: str (CSV file path)
        
        -   Path to the `books_matched.csv` file within `DATA_PATH`.
            
    -   `MAX_FILE_SIZE_MB`: int
        
        -   Maximum desired size for each chunked output file in megabytes.

    -   `OUTPUT_PATH`: str (directory)
        
        -   The directory (`split_data/`) where all generated split and chunked CSV files are saved.            
     
-   **Returns:**
    
    -   `split_data/train_ratings.csv`: CSV file(s)
        
        -   The primary training set for user ratings. Also available as chunked files (e.g., `train_ratings_part_1.csv`).
            
    -   `split_data/cross_val_ratings.csv`: CSV file(s)
        
        -   The cross-validation set for user ratings. Also available as chunked files (e.g., `cross_val_ratings_part_1.csv`).
            
    -   `split_data/test_ratings.csv`: CSV file(s)
        
        -   The testing set for user ratings. Also available as chunked files (e.g., `test_ratings_part_1.csv`).
            
    -   `split_data/books_train.csv`: CSV file(s)
        
        -   The training set for book data, matched with the user rating splits. Also available as chunked files (e.g., `books_train_part_1.csv`).
            

**4. KEYWORD PREPROCESSING (optional):** 
Create a list of potential keywords from book descriptions by removing common and problematic words. Reformat book descriptions to ensure consistency.

-   `keyword_preprocessing\keyword_preprocessing.py`
    
-   Takes words from books' descriptions and users' reviews, formats them consistently, removes special characters and numerals, and removes short words and words commonly found in book reviews (stopwords).
    
-   **Parameters:**
    
    -   `books_matched_copy.csv`: str (CSV file)
        
        -   Contains an indexed list of books, where each book has a description (`description`).
            
    -   `reviews_matched_copy.csv`: str (CSV file)
        
        -   Contains an indexed list of user reviews of books, where each user review may have user-written text (`review`).
            
    -   `new_stop_words`: str list
        
        -   Additional stopwords to add to the default list of stopwords from the `nltk.corpus` library. Generally, contains words common in book descriptions and reviews, such as words related to reading, publishing, or writing.
            
-   **Returns:**
    
    -   `output_file_books.csv`: CSV file
        
        -   Contains all potential keywords from each book description from the `books_matched_copy.csv` dataset.
            
    -   `output_file_reviews.csv`: CSV file
        
        -   Contains all potential keywords from each user review from the `reviews_matched_copy.csv` dataset. Not as useful since book reviews are usually blank (ratings only).


**5. KEYWORD EXTRACTION & PROCESSING:**
The following scripts are used to extract keywords from book descriptions, cluster them semantically, and then process these keywords into numerical vectors for analysis.

-   `keyword_analysis_keyBERT\keyword_extraction_final.py`
    
-   Creates a list of keywords from each book's description using the KeyBERT model. Handles text preprocessing, keyword extraction, and saving the results.
    
-   **Parameters:**
    
    -   `data_folder_path`: str

        -   The path to the directory containing the input CSV files (e.g., `'data_folder'`). All CSV files found within this folder will be processed.

    -   `n_values`: list of int

        -   A list of integer values representing the `top_n` parameter for KeyBERT (maximum number of keywords to extract per document). The script will run keyword extraction for each `n` value in this list (e.g., `[15, 10]`).
        
    -   `diversity_values`: list of float

        -   A list of float values representing the `diversity` parameter for KeyBERT, controlling the uniqueness of extracted keywords. The script will run keyword extraction for each `diversity` value in this list (e.g., `[0.2, 0.4]`).
    
-   **Returns:**
    
    -   Output CSV file(s) : CSV file

        -   CSV file(s) containing the original book metadata, descriptions (raw and preprocessed), and KeyBERT-extracted keywords, with filename(s) dynamically reflecting the input source and parameters.

-   `keyword_clustering\cluster_keywords.py`

-   Extracts keywords into semantic clusters and facilitates the creation of keyword feature vectors.

-   **Parameters:**

    -   `INPUT_FILE_PATH`: str

        -   The path to the input CSV file containing a 'keywords' column (default: `../keyword_analysis_keyBERT/output_file_books_train_preprocessed_n15_div0.2.csv`).

    -   `OUTPUT_FILE_PATH`: str

        -   The path to save the output CSV file with keyword clusters (default: `keys_by_cluster.csv`).

    -   `MIN_COMMUNITY_SIZE`: int

        -   The minimum number of elements required for a cluster to be considered valid (default: `4`).

    -   `THRESHOLD`: float

        -   The cosine-similarity threshold used by the community detection algorithm. Keyword pairs with a similarity larger than this threshold are considered similar and potentially grouped (default: `0.65`).

-   **Returns:**

    -   `keys_by_cluster.csv`: CSV file

        -   Contains a mapping of each identified cluster to a comma-separated list of keywords belonging to that cluster. 

-   `keyword_processing\get_keys_v2.py`
    
-  This script focuses on transforming extracted keywords and genres into numerical feature vectors based on predefined keyword clusters.
    
-   **Parameters:**
    
    -   `key_clusters_filepath`: str (CSV file path)
        
        -   Path to the CSV file containing keyword cluster definitions (e.g., `../keyword_clustering/keys_by_cluster.csv`, which is an output from `cluster_keywords.py`). This file defines which keywords belong to which clusters.
            
    -   `books_filepath`: str (CSV file path)
        
        -   Path to the CSV file containing preprocessed book data, which *must* include 'genres' and 'keywords' columns (e.g., the output from `keyword_extraction_final.py` like `../keyword_analysis_keyBERT/output_file_books_train_preprocessed_n15_div0.2.csv`).
            
-   **Returns:**
    
    -   `books_with_vectors.csv`: CSV file
        
        -   Includes book titles, authors, ratings, and number of ratings, along with generated binary keyword vectors (and their sums) and genre vectors (and their sums).


**6. K-MODES BOOK CLUSTERING (optional):** 
Create clusters of similar books using their feature vectors and the k-modes algorithm (a variant of k-modes for categorical data). Use these clusters to reduce runtime when making book recommendations.

-   `k_means_clustering\k_modes_v2.py`
    
-   Generates a fixed number of book clusters using k-modes, using feature vectors of genres and (clustered) keywords for each book. This updated version uses custom dissimilarity metrics with different weights given to keywords and genres.
    
-   **Parameters:**
    
    -   `num_clusters`: int
        
        -   The predetermined number of clusters to form using KModes.
            
    -   `cat_dissim`: callable
        
        -   Specifies the function to use as the dissimilarity metric (e.g., `hamming_dist`, `cosine_dissim`).
            
    -   `books_with_vectors_and_sums_and_ratings.csv`: str (CSV file)
        
        -   CSV file containing books (titles and authors) and their feature vectors. Two separate feature vectors for genres and (clustered) keywords are generally provided. Also includes number of ratings and average rating for each book.
            
    -   `keyword_weight`: float
        
        -   How much keywords should be weighted relative to genres when measuring similarities. The optimal is somewhere around 0.8.
            
    -   Additional parameters used by KModes (e.g., `init`, `n_init`, `max_iter`, `verbose`).
        
-   **Returns:**
    
    -   `books_clustered.csv`: CSV file
        
        -   The input CSV file with an additional column indicating the assigned cluster for each book.
            
    -   `books_by_cluster.csv`: CSV file
        
        -   CSV file containing lists of books, grouped by their assigned cluster.
            
-   `k_means_clustering\elbow_method_v2.py`
    
-   Determines optimal number of book clusters for k-modes, using feature vectors of genres and (clustered) keywords for each book. This updated version uses custom dissimilarity metrics with different weights given to keywords and genres.
    
-   **Parameters:**
    
    -   `K`: int list
        
        -   List of numbers of initial cluster centroids to test with KModes.
            
    -   `cat_dissim`: callable
        
        -   Specifies the function to use as the dissimilarity metric (e.g., `hamming_dist`, `cosine_dissim`).
            
    -   `books_with_vectors_and_sums.csv`: str (CSV file)
        
        -   CSV file containing books (titles and authors) and their feature vectors. Two separate feature vectors for genres and (clustered) keywords are generally provided.
            
    -   `keyword_weight`: float
        
        -   How much keywords should be weighted relative to genres when measuring similarities. The optimal is somewhere around 0.8.
            
    -   Additional parameters used by KModes (e.g., `init`, `n_init`, `max_iter`, `verbose`).
        
-   **Returns:**
    
    -   `elbow_graph.png`: PNG file
        
        -   The bend in the elbow plot is approximately the optimal number of book clusters in terms of costs.
            
-   `k_means_clustering\small_read_list_recommendations_v1.py`
    
-   Given a list of books that a user has read and enjoyed (no ratings are required to be specified), generates a list of popular books that share a k-modes cluster with any book in the list. Designed to be a simpler recommendation system for readers who want to provide a shorter list of books they have read.
    
-   **Parameters:**
    
    -   `sample_user_books`: str list
        
        -   An unrated list of books (titles and authors) that a user has read before.
            
    -   `books_clustered_ratings.csv`: str (CSV file)
        
        -   CSV file containing books (titles and authors) and their feature vectors, with an additional column indicating the assigned cluster for each book. Also includes number of ratings and average rating for each book.
            
    -   `books_by_cluster_ratings.csv`: str (CSV file)
        
        -   CSV file containing lists of books, grouped by their assigned cluster. Must correspond to the clusters in `books_clustered_ratings.csv`.
            
    -   `std_num_outlier`: float
        
        -   The number of standard deviations below the mean similarity that serves as the threshold for outliers.
            
    -   `highest_hits_to_keep`: int
        
        -   The number of books with the highest number of ratings to keep. Should be very high relative to the number of books left after filtering for outliers.
            
    -   `highest_ratings_to_keep`: int
        
        -   The number of the most highly rated books (in terms of average rating) to keep. Should be less than `highest_hits_to_keep`.
            
    -   `keyword_weight`: float
        
        -   How much keywords should be weighted relative to genres when measuring similarities. The optimal is somewhere around 0.8.
            
-   **Returns:**
    
    -   `recommendations.csv`: CSV file
        
        -   Popular books that share a k-modes cluster with a book the user has read before, sorted by average rating. 


**7A. CONTENT-BASED FILTERING:** 
Train the content-based filtering model.

-   `content_filtering\train_similarity_final.py`
    
-   Reads book data with keyword and genre vectors, calculates a combined cosine similarity matrix based on configurable weights, and exports the matrix and corresponding book identifiers.
    
-   **Parameters:**
    
    -   `books_with_vectors_filepath`: str (CSV file path)
        
        -   Path to the CSV file containing book titles, authors, and keyword/genre vectors as string representations of lists (e.g., `../keyword_processing/books_with_vectors.csv` from Keyword Processing).
            
    -   `weight_pairs`: list of tuple
        
        -   Weights to assign to genre and keyword similarity in the combined similarity calculation. Defaults to `[(0.8, 0.2)]`.
            
    -   `checkpoint_dir`: str, optional
        
        -   Directory path for saving and loading intermediate similarity matrix checkpoints (`.npy` files). Defaults to `./similarity_checkpoints`.
            
-   **Returns:**
    
    -   `book_similarity_matrix_{genre_weight_normalized}.npy`: NumPy `.npy` file
        
        -   A square NumPy array representing the combined similarity matrix between all books. The filename will include the genre weight (e.g., `book_similarity_matrix_0-8.npy`).
            
    -   `book_similarity_matrix_books_{genre_weight_normalized}.txt`: Text file
        
        -   A text file containing a list of book identifiers ("title, author name"), corresponding to the order of books in the similarity matrix. The filename will include the normalized genre weight.
            
    -   (Implicit) Intermediate Checkpoints:
        
        -   `keyword_sim_matrix_gX-X_kX-X.npy`: NumPy `.npy` file (saved in `checkpoint_dir`)
            
            -   Intermediate checkpoint for the keyword similarity matrix.
                
        -   `genre_sim_matrix_gX-X_kX-X.npy`: NumPy `.npy` file (saved in `checkpoint_dir`)
            
            -   Intermediate checkpoint for the genre similarity matrix.
                
-   `content_filtering\numpy_to_h5.py`
    
-   Converts a large NumPy array from a `.npy` file into a HDF5 (`.h5`) file format.
    
-   **Parameters:**
    
    -   `ORIGINAL_NPY_FILEPATH`: str
        
        -   The path to the input NumPy `.npy` file (e.g., `book_similarity_matrix_0-8.npy` generated by `train_similarity_final.py`).
            
    -   `OUTPUT_H5_FILEPATH`: str
        
        -   The desired path for the output HDF5 `.h5` file (e.g., `book_similarity_matrix.h5`).
            
    -   `H5_DATASET_NAME`: str
        
        -   The name of the dataset within the HDF5 file where the NumPy array will be stored (e.g., `similarity_matrix`).
            
-   **Returns:**
    
    -   `{OUTPUT_H5_FILEPATH}`: HDF5 (.h5) file
        
        -   An HDF5 file containing the converted NumPy array as a dataset. 
            
-   `content_filtering\split_h5.py`
    
-   Splits a large HDF5 file into smaller HDF5 chunks.
   
    -   `LARGE_H5_FILEPATH`: str
        
        -   Path to the original large HDF5 file (e.g., `book_similarity_matrix.h5`).
            
    -   `LARGE_H5_DATASET_NAME`: str
        
        -   The name of the dataset within the HDF5 file to be split (e.g., `similarity_matrix`).
            
    -   `CHUNK_SIZE_ROWS`: int
        
        -   The maximum desired number of rows for each output HDF5 chunk file (e.g., `3000`).
            
    -   `OUTPUT_DIR`: str
        
        -   The directory where the chunked HDF5 files will be saved (e.g., `split_h5_chunks`).
            
-   **Returns:**
    
    -   `{OUTPUT_DIR}/chunk_XXX.h5`: HDF5 (.h5) files
        
        -   Multiple HDF5 files, each representing a chunk of the original dataset. These files are saved in the specified `OUTPUT_DIR` and named sequentially (e.g., `chunk_000.h5`, `chunk_001.h5`). Each chunk file contains a dataset named `chunk_data_XXX`.
            

**7B. COLLABORATIVE FILTERING:** 
Train the collaborative filtering model.

-   `collab_filtering\cf_final.py`
    
-   Trains a collaborative filtering recommendation system using matrix factorization and exports the trained model.
    
-   **Parameters:**

    -   `DATA_PATH`: str 

        -   The base directory where the split ratings files are located (e.g., `../train_test_split/split_data_combined/`).
    
    -   `TRAIN_RATINGS_FILE`: str 

        -   The full path to the training ratings CSV file (e.g., `../train_test_split/split_data_combined/train_ratings.csv`).
    
    -   `CROSS_VAL_RATINGS_FILE`: str 

        -   The full path to the cross-validation ratings CSV file (e.g., `../train_test_split/split_data_combined/cross_val_ratings.csv`).
            
    -   `LEARNING_RATE`: float

        -   The learning rate used for the gradient descent optimization in matrix factorization (e.g., `0.005`).
            
    -   `ITERATIONS`: int

        -   The number of training iterations for the matrix factorization model (e.g., `1000`).
            
    -   `TOP_N_RECOMMENDATIONS`: int

        -   The default number of top recommendations to generate (e.g., `20`).
            
    -   `TRAINED_DATA_FOLDER`: str

        -   The directory where trained model artifacts and checkpoints will be saved (e.g., `trained_data/`).
            
    -   `TRAINING_CHECKPOINT_FILENAME`: str

        -   The full path and filename for saving training progress checkpoints (e.g., `trained_data/training_progress_checkpoint.pkl`).
            
    -   `NUM_FEATURES`: int

        -   The number of latent features to learn for users and items in the matrix factorization model (e.g., `600`).
            
    -   `LAMBDA_VAL`: int or float

        -   The regularization parameter (lambda) used to prevent overfitting in the matrix factorization model (e.g., `5`).
                    
-   **Returns:**
         
    -   `X_features_{num_features}_lambda_{lambda_val}.npy`: NumPy `.npy` file (saved in `output_dir`)
        
        -   Learned item features (`X`) from the matrix factorization model.
            
    -   `b_item_bias_{num_features}_lambda_{lambda_val}.npy`: NumPy `.npy` file (saved in `output_dir`)
        
        -   Learned item biases (`b_item`) from the matrix factorization model.
            
    -   `book_list_features_{num_features}_lambda_{lambda_val}.txt`: Text file (saved in `output_dir`)
        
        -   A list of unique book identifiers used in the trained model.
            
    -   `Y_mean_features_{num_features}_lambda_{lambda_val}.npy`: NumPy `.npy` file (saved in `output_dir`)
        
        -   The mean rating for each book used in the model.
            

**7C. COMBINED FILTERING:** 
Generate book recommendation lists using a combination of content-based filtering and collaborative filtering.

-   `combined_model\cb_get_recs_new.py`
    
-   This script calculates content-based recommendations for a user based on their rated books. It utilizes pre-computed similarity data stored in HDF5 chunks to compute weighted similarity scores for all books.
    
-   **Parameters:**
    
    -   `BASE_FILE_NAME`: str

        -   The base name for the similarity matrix and book identifiers text file (e.g., `../content_filtering/book_similarity_matrix`).
            
    -   `GENRE_WEIGHT`: float

        -   The genre weight used in the combined similarity calculation, which is used to derive the `BOOK_IDENTIFIERS_TXT_FILEPATH` (e.g., `0.8`).
            
    -   `H5_CHUNKS_DIR`: str

        -   The directory containing the chunked HDF5 similarity matrices (e.g., `../content_filtering/split_h5_chunks`).
            
    -   `H5_CHUNK_FILE_PREFIX`: str

        -   The prefix used for the filenames of the HDF5 chunk files (e.g., `chunk_`).
            
    -   `H5_CHUNK_DATASET_PREFIX`: str

        -   The prefix used for the dataset names *inside* the HDF5 chunk files (e.g., `chunk_data_`).
            
    -   `CHUNK_SIZE_ROWS`: int

        -   The number of rows per chunk used during the initial HDF5 file splitting. This value is crucial for correctly processing the chunks (e.g., `3000`).
            
    -   `BOOK_IDENTIFIERS_TXT_FILEPATH`: str

        -   The full path to the text file containing the book identifiers, derived from `BASE_FILE_NAME` and `GENRE_WEIGHT` (e.g., `../content_filtering/book_similarity_matrix_books_0-8.txt`). This file corresponds to the order of books in the HDF5 similarity matrix.
            
    -   `user_books`: list of str

        -   A list of book identifiers ("title, author name") that the user has rated.
            
    -   `user_ratings`: list of int or float

        -   A list of corresponding ratings for `user_books` (e.g., 1-5 scale).
            
-   **Returns:**
    
    -   `pd.DataFrame`:
        
        -   A DataFrame with 'Book Identifier', 'Weighted Similarity Score', 'Rated by User' (boolean), and 'Original Rating' columns. Returns an empty DataFrame if no valid rated books are found or an error occurs.

-   `combined_model\cf_get_recs.py`
    
-   Prepares user ratings, trains a user-specific Ridge regression model using pre-trained book features and biases, and calculates predicted ratings for all books. The output includes user-rated books with their original ratings and a flag indicating if they were rated.
    
-   **Parameters:**
    
    -   `book_features`: np.ndarray
        
        -   The full matrix of all book features.
            
    -   `item_bias`: np.ndarray
        
        -   The full item bias vector.
            
    -   `mean_book_ratings`: np.ndarray
        
        -   The mean rating for all books.
            
    -   `all_book_identifiers`: list[str]
        
        -   The master list of all book identifiers.
            
    -   `total_books`: int
        
        -   Total number of books.
            
    -   `user_rated_book_identifiers`: list[str]
        
        -   List of book identifiers rated by the user.
            
    -   `user_book_ratings`: list[float]
        
        -   List of ratings corresponding to user_rated_book_identifiers.
            
    -   `new_user_regularization_strength`: float
        
        -   Regularization parameter for the new user model.
            
-   **Returns:**
    
    -   `pd.DataFrame` or `None`: Pandas DataFrame with 'Book Identifier', 'Predicted Rating', 'Rated by User', and 'Original Rating' columns, sorted by 'Predicted Rating' in descending order. Returns `None` if user model training fails.
        
-   `combined_model\combined_get_recs_final.py`
    
-   Generates combined book recommendations using both content-based similarity and collaborative filtering predictions. It loads necessary models and data, then applies multiple strategies (fractional, filtered by CF, multiplied scores, and hybrid) to generate and display recommendation lists.
    
-   **Parameters:**
    
    -   `user_books`: list[str]
        
        -   A list of book identifiers ("title, author name") that the user has rated.
            
    -   `user_ratings`: list[float]
        
        -   A list of corresponding ratings (e.g., 1-5 scale) for `user_books`.
            
    -   `genre_weight`: int, optional
        
        -   The weight applied to genre similarity in content-based model file paths. Defaults to 0.8.
            
    -   `num_features`: int, optional
        
        -   Number of latent features for the collaborative filtering model. Defaults to 600.
            
    -   `new_user_regularization_strength`: int, optional
        
        -   Regularization strength of a new user for the collaborative filtering model. Defaults to 5.
            
-   **Returns:**
    
    -   `pd.DataFrame` or `None`:
        
        -   The primary return is a DataFrame containing:
            
            -   `'Book Identifier'`: Combined book identifier ("title, author name").
                
            -   `'CB_Weighted_Similarity_Score'`: Content-based similarity score.
                
            -   `'CF_Predicted_Rating'`: Collaborative filtering predicted rating.
                
            -   `'Rated by User'`: Boolean indicating if the book was rated by the user.
                
            -   `'Original Rating'`: The user's original rating for the book (NaN if not rated).
            
    -   (Implicit) Printed Recommendation Lists:
        
        -   The script also directly prints various recommendation lists to the console based on different strategies:
            
            -   **Strategy 1: CB Top N Filtered by CF:** Recommends books by first taking the top N from content-based recommendations, then filtering and re-ranking that list based on collaborative filtering scores.
                
            -   **Strategy 2: Scores Multiplied Together:** Recommends books by creating a new combined score by multiplying the content-based similarity score and the collaborative filtering predicted rating.
                
            -   **Strategy 3: Pure CB:** Recommends a specified fraction of the top N from content-based recommendations.
                
            -   **Strategy 4: Hybrid Strategy 2+1:** Recommends books by combining a fraction of pure top N content-based recommendations with the remaining fraction from a list of top content-based books re-ranked by collaborative filtering, prioritizing books appearing in both lists.
                

**8. TESTING AND VALIDATION:** 
Verify the accuracy of book recommendation lists.

-   `combined_model\testing_final.py`
    
-   This script manages a recommendation evaluation pipeline for a predefined configuration. It loads, preprocesses, and splits user rating data into training and validation sets. Recommendations are then generated for users using various strategies (content-based, collaborative filtering, hybrid, and random baseline). The script evaluates performance by calculating the overlap of recommendations with relevant validation books. It includes a checkpointing system for resumption, saving intermediate DataFrames as Parquet files and final evaluation results as a CSV. Core recommendation logic and helpers are imported from `combined_recs_testing_final.py`.
    
-   **Internal Configuration Parameters:**
    
    -   The script defines its operational parameters internally within a `test_config` dictionary and other variables. These values directly influence the behavior of the imported recommendation functions:
        
    -   `user_ratings_file_path`: Located at `../train_test_split/split_data_combined/test_ratings.csv`. This path specifies the input CSV file containing raw user ratings.
        
    -   `relevance_threshold`: Set to `3.0`. This minimum rating determines if a book is considered "relevant" in the validation set for evaluation.
        
    -   `validation_split_ratio`: Set to `0.5`. This is the fraction of a user's rated books reserved for the validation set.
        
    -   `genre_weight`: Configured as `0.8`. This weight is applied to genre similarity in content-based model file paths.
        
    -   `num_features`: Configured as `600`. This specifies the number of latent features for the collaborative filtering model.
        
    -   `new_user_regularization_strength`: Configured as `5` (from `lambda_val`). This is the regularization strength for the new user collaborative filtering model.
        
    -   `output_limit_per_strategy`: Set to `60`. This is the maximum number of recommendations generated for each strategy for each user.
        
-   **Outputs:**
    
    -   **Intermediate Parquet Files:** The script saves various intermediate DataFrames (e.g., processed user data, training/validation splits, generated recommendations, relevant validation books) as `.parquet` files within a configuration-specific subdirectory under `single_run_evaluation_results_loop/`.
        
    -   **Final Overlap Evaluation CSV:** A CSV file named using the configuration parameters (e.g., `overlap_evaluation_nf600_rs5_gw0-8.csv`) is saved to the configuration's output directory. This DataFrame contains the final evaluation results with the following columns:
        
        -   `'User_id'`: The ID of the user.
            
        -   `'Total_Relevant_Books'`: The total count of relevant books in the user's validation set.
            
        -   `'Overlap_Rec_CB_Only'`: The number of recommended books from the pure content-based strategy that overlap with the user's relevant validation books.
            
        -   `'Overlap_Rec_CF_Only'`: The number of recommended books from the pure collaborative filtering strategy that overlap with the user's relevant validation books.
            
        -   `'Overlap_Rec_CB_Filtered_by_CF'`: The number of recommended books from the content-based strategy (re-ranked by CF) that overlap with the user's relevant validation books.
            
        -   `'Overlap_Rec_By_Multiplying_Scores'`: The number of recommended books from the multiplied scores strategy that overlap with the user's relevant validation books.
            
        -   `'Overlap_Rec_Random'`: The number of recommended books from the random strategy that overlap with the user's relevant validation books.
            
-   `combined_model\analyze.py`
    
-   This script handles the statistical analysis and visualization of the recommendation overlap results generated by `testing_final.py`. It loads the `overlap_evaluation` CSV files, calculates descriptive statistics (mean, median, skew, etc.) for overall performance and within different bins of "total relevant books" (e.g., low, medium, high), and then generates various plots (bar charts, KDE plots) to help interpret the effectiveness of different recommendation strategies. It is the designated visualization component of the pipeline.
    
-   **Parameters:**
    
    -   `CHECKPOINT_ROOT_DIR`: str
        
        -   The root directory where the output directories from `testing_final.py` are located (e.g., `single_run_evaluation_results_loop`).
            
    -   `parameters_to_process`: list of dict
        
        -   A list of dictionaries, where each dictionary specifies the configuration (e.g., `{'num_features': 600, 'lambda_val': 5, 'genre_weight': 0.8}`) for which `overlap_evaluation` CSV files are to be analyzed.
            
    -   `OUTPUT_RESULTS_FILE`: str, optional
        
        -   The path to a text file where the detailed statistical analysis results will be written. If `None`, results are only printed to the console.
            
-   **Returns:**
    
    -   `{OUTPUT_RESULTS_FILE}`: Text file
        
        -   A text file containing the detailed descriptive statistics for each processed overlap evaluation file, including overall and binned results.
            
    -   Plot Files (PNG):
            
        -   `overall_mean_overlap_nf{nf}_rs{rs}_gw{gw}.png`
                
        -   `binned_mean_overlap_nf{nf}_rs{rs}_gw{gw}.png`
                
        -   `distribution_{strategy_name_lower}_nf{nf}_rs{rs}_gw{gw}.png` (e.g., `distribution_overlap_rec_cb_only_nf600_rs5_gw0.8.png`)
                
-   **MANUAL TESTING**
    
-   The quality of book recommendation lists can be verified by developers by inputting a list of books with ratings out of five stars into `combined_model\combined_get_recs_final.py` and then evaluating the quality of the resulting recommendations.


**APP:**

-   `app\app.py`

-   Creates an app with a user interface that makes the recommendation process easier. 

-   `app\app_v2.py`

-   Creates an app with a user interface that makes the recommendation process easier. Includes a k-modes recommender.

-   `k_means_clustering\small_read_list_recommendations_v2.py`

-   Version of `k_means_clustering\small_read_list_recommendations_v1.py` that runs with the app `app\app_v2.py`.


**OTHER IMPORTANT SCRIPTS:** 

-   `large_files.py` (various locations) 

-   Breaks large CSV files into smaller chunks so that GitHub can store them properly.

-   `recombine_files.py` (various locations) 

-   Recombines file chunks created by `large_files.py` into one CSV file. 

-   `big_data_set\combine_v2.py`

-   Recombines file chunks created by `large_files.py` into one CSV file. Newer version?

-   `cleaning\create_fake_data.py` 

-   Generates fake feature vectors for a list of title-author pairs (for testing purposes). 

-   `collab_filtering\create_dummy_data.py`

-   Generates fake data (for testing purposes)?

-   `fake_data\genre_list.py`

-   Searches for books by genre?

-   `big_data_set\find_ids.py`

-   Finds user IDs?


**OTHER SCRIPTS NOT USED:**
-   `big_data_set\scripts_older_versions\author_names.py`
-   `big_data_set\scripts_older_versions\combine_v1.py` 
-   `cleaning\scripts_older_versions\goodreads_ratings_change_df.py` 
-   `cleaning\scripts_older_versions\goodreads_ratings_change_df.ipynb`
-   `cleaning\scripts_older_versions\same_books_v1.py` 
-   `collab_filtering\scripts_older_versions\cf_new_user_v1.py` 
-   `collab_filtering\scripts_older_versions\cf_new_user_v2.py` 
-   `collab_filtering\scripts_older_versions\cf_new_user_v3.py` 
-   `collab_filtering\scripts_older_versions\cf_new_user_v4.py` 
-   `collab_filtering\scripts_older_versions\cf_new_user_v5.py` 
-   `collab_filtering\scripts_older_versions\cf_silly.py` 
-   `collab_filtering\scripts_older_versions\cf_v1.py` 
-   `collab_filtering\scripts_older_versions\cf_v2.py` 
-   `collab_filtering\scripts_older_versions\cf_v3.py` 
-   `collab_filtering\scripts_older_versions\cf_v4.py` 
-   `collab_filtering\scripts_older_versions\cf_v5.py` 
-   `collab_filtering\scripts_older_versions\cf_v6.py` 
-   `collab_filtering\scripts_older_versions\cf_v7.py` 
-   `combined_model\scripts_older_versions\analyze_old.py`
-   `combined_model\scripts_older_versions\bagging_test.py`
-   `combined_model\scripts_older_versions\combined_recs_testing_v1.py` 
-   `combined_model\scripts_older_versions\combined_recs_testing_v2.py` 
-   `combined_model\scripts_older_versions\testing_no_print_v1.py` 
-   `combined_model\scripts_older_versions\testing.py` 
-   `content_filtering\scripts_older_versions\get_recs.py` 
-   `content_filtering\scripts_older_versions\train_similarity_v1.py` 
-   `k_means_clustering\scripts_older_versions\elbow_method_v1.py` 
-   `k_means_clustering\scripts_older_versions\k_modes_v1.py` 
-   `keyword_analysis_keyBERT\scripts_older_versions\keyword_extraction_v1.py` 
-   `keyword_analysis_keyBERT\scripts_older_versions\keyword_extraction_v2.py` 
-   `keyword_processing\scripts_older_versions\get_keys_v1.py`