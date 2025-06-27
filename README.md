# Goodreaders

[](https://github.com/catmj/Goodreaders#goodreaders)

Goodreaders book recommender project for The Erdos Institute Data Science Bootcamp. Catherine Johnston, Yevgeniya “Jonah” Tarasova, Katie Johnston, and Ava Bamforth

Procedural order from data collection to book recommendations and verification: 


**1. SCRAPING:** 
Collect data from "https://www.goodreads.com/" using BeautifulSoup web scraping. Scrape data for individual books, as well as data for individual user reviews of books.

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

- Parameters:

"reviews_with_urls_left_to_scrape.csv" : str (CSV file)

Each row corresponds to a book and contains the portion of its URL after "https://www.goodreads.com/book/show/". Some books have duplicate URLs, particularly books whose titles contain non-Latin characters.

- Returns:

"books_in_progress.csv" : CSV file

Adds a row of data for each book scraped, one at a time. Can add rows to an existing file "books_in_progress.csv".

**2. DATA CLEANING AND MATCHING:** 

- cleaning\same_books_v2.py

- Modify scraped data to make formats more consistent, remove problematic books, etc. Match user reviews of books with the corresponding books.
  

- **Parameters:**

- **Returns:**

- 
            

**3. DATA SPLITTING:**

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
            
-   **Returns:**
    
    -   `split_data/train_ratings.csv`: CSV file(s)
        
        -   The primary training set for user ratings. Also available as chunked files (e.g., `train_ratings_part_1.csv`).
            
    -   `split_data/cross_val_ratings.csv`: CSV file(s)
        
        -   The cross-validation set for user ratings. Also available as chunked files (e.g., `cross_val_ratings_part_1.csv`).
            
    -   `split_data/test_ratings.csv`: CSV file(s)
        
        -   The testing set for user ratings. Also available as chunked files (e.g., `test_ratings_part_1.csv`).
            
    -   `split_data/books_train.csv`: CSV file(s)
        
        -   The training set for book data, matched with the user rating splits. Also available as chunked files (e.g., `books_train_part_1.csv`).
            
    -   `OUTPUT_PATH`: Path (directory)
        
        -   The directory (`split_data/`) where all generated split and chunked CSV files are saved.
            

**4. KEYWORD PREPROCESSING (optional):** Create a list of potential keywords from book descriptions by removing common and problematic words. Reformat book descriptions to ensure consistency.

-   `keyword_preprocessing\keyword_preprocessing.py`
    
-   Takes words from books' descriptions and users' reviews, formats them consistently, removes special characters and numerals, and removes short words and words commonly found in book reviews (stopwords).
    
-   **Parameters:**
    
    -   "books_matched_copy.csv" : str (CSV file)
        
        -   Contains an indexed list of books, where each book has a description ("description").
            
    -   "reviews_matched_copy.csv" : str (CSV file)
        
        -   Contains an indexed list of user reviews of books, where each user review may have user-written text ("review").
            
    -   new_stop_words : str list
        
        -   Additional stopwords to add to the default list of stopwords from the `nltk.corpus` library. Generally, contains words common in book descriptions and reviews, such as words related to reading, publishing, or writing.
            
-   **Returns:**
    
    -   "output_file_books.csv" : CSV file
        
        -   Contains all potential keywords from each book description from the "books_matched_copy.csv" dataset.
            
    -   "output_file_reviews.csv" : CSV file
        
        -   Contains all potential keywords from each user review from the "reviews_matched_copy.csv" dataset. Not as useful since book reviews are usually blank (ratings only).
            

**5. KEYWORD PROCESSING:** Create a list of keywords from each book's description.

-   `keyword_processing\get_keys_v2.py`
    
-   This script handles the loading, preprocessing, and vector generation for book data. It reads keyword cluster information, loads and cleans raw book data (extracting genres and keywords), generates numerical feature vectors for both keywords (based on clusters) and genres, and then exports the processed book data along with these vectors and their sums to a CSV file.
    
-   **Parameters:**
    
    -   `key_clusters_filepath`: str (CSV file path)
        
        -   Path to the CSV file containing keyword clusters (e.g., `../keyword_clustering/keys_by_cluster.csv`).
            
    -   `books_filepath`: str (CSV file path)
        
        -   Path to the CSV file containing preprocessed book data, including 'genres' and 'keywords' columns (e.g., `../keyword_analysis_keyBERT/output_file_books_train_preprocessed_n15_div0.2.csv` from Keyword Processing).
            
-   **Returns:**
    
    -   `books_with_vectors.csv`: CSV file
        
        -   Contains book titles, authors, `rating`, `num_ratings`, the generated `keyword_vector` (list representation of a binary vector indicating keyword cluster presence), and `keyword_vector_sum` (sum of elements in the keyword vector).
            
        -   Also contains `genre_vector` (list representation of a binary vector indicating genre presence), and `genre_vector_sum` (sum of elements in the genre vector).
            
        -   This file is sorted by `keyword_vector_sum` then `genre_vector_sum` in descending order.
            

**6. KEYWORD CLUSTERING:** Create clusters of keywords that have similar meanings. Assign each cluster a component in the feature vectors.

-   `keyword_clustering\cluster_keywords.py`
    
-   Loads keywords, filters out non-English words and short words, then uses Sentence-BERT embeddings and community detection to group semantically similar keywords into clusters.
    
-   **Parameters:**
    
    -   `../keyword_analysis_keyBERT/output_file_books_train_preprocessed_n15_div0.2.csv`: str (CSV file path)
        
        -   The input CSV file containing book keywords that need to be clustered.
            
-   **Returns:**
    
    -   `keys_by_cluster.csv`: CSV file
        
        -   Contains a list of keyword clusters, where each row represents a cluster and contains the keywords belonging to that cluster, separated by commas.
            

**7. K-MODES BOOK CLUSTERING (optional):** Create clusters of similar books using their feature vectors and the k-modes algorithm (a variant of k-modes for categorical data). Use these clusters to reduce runtime when making book recommendations.

-   `k_means_clustering\k_modes_v2.py`
    
-   Generates a fixed number of book clusters using k-modes, using feature vectors of genres and (clustered) keywords for each book. This updated version uses custom dissimilarity metrics with different weights given to keywords and genres.
    
-   **Parameters:**
    
    -   num_clusters : int
        
        -   The predetermined number of clusters to form using KModes.
            
    -   cat_dissim : callable
        
        -   Specifies the function to use as the dissimilarity metric (e.g., `hamming_dist`, `cosine_dissim`).
            
    -   "books_with_vectors_and_sums_and_ratings.csv" : str (CSV file)
        
        -   CSV file containing books (titles and authors) and their feature vectors. Two separate feature vectors for genres and (clustered) keywords are generally provided. Also includes number of ratings and average rating for each book.
            
    -   keyword_weight : float
        
        -   How much keywords should be weighted relative to genres when measuring similarities. The optimal is somewhere around 0.8.
            
    -   Additional parameters used by KModes (e.g., `init`, `n_init`, `max_iter`, `verbose`).
        
-   **Returns:**
    
    -   "books_clustered.csv" : CSV file
        
        -   The input CSV file with an additional column indicating the assigned cluster for each book.
            
    -   "books_by_cluster.csv" : CSV file
        
        -   CSV file containing lists of books, grouped by their assigned cluster.
            
-   `k_means_clustering\elbow_method_v2.py`
    
-   Determines optimal number of book clusters for k-modes, using feature vectors of genres and (clustered) keywords for each book. This updated version uses custom dissimilarity metrics with different weights given to keywords and genres.
    
-   **Parameters:**
    
    -   K : int list
        
        -   List of numbers of initial cluster centroids to test with KModes.
            
    -   cat_dissim : callable
        
        -   Specifies the function to use as the dissimilarity metric (e.g., `hamming_dist`, `cosine_dissim`).
            
    -   "books_with_vectors_and_sums.csv" : str (CSV file)
        
        -   CSV file containing books (titles and authors) and their feature vectors. Two separate feature vectors for genres and (clustered) keywords are generally provided.
            
    -   keyword_weight : float
        
        -   How much keywords should be weighted relative to genres when measuring similarities. The optimal is somewhere around 0.8.
            
    -   Additional parameters used by KModes (e.g., `init`, `n_init`, `max_iter`, `verbose`).
        
-   **Returns:**
    
    -   "elbow_graph.png" : PNG file
        
        -   The bend in the elbow plot is approximately the optimal number of book clusters in terms of costs.
            
-   `k_means_clustering\small_read_list_recommendations.py`
    
-   Given a list of books that a user has read and enjoyed (no ratings are required to be specified), generates a list of popular books that share a k-modes cluster with any book in the list. Designed to be a simpler recommendation system for readers who want to provide a shorter list of books they have read.
    
-   **Parameters:**
    
    -   sample_user_books : str list
        
        -   An unrated list of books (titles and authors) that a user has read before.
            
    -   "books_clustered_ratings.csv" : str (CSV file)
        
        -   CSV file containing books (titles and authors) and their feature vectors, with an additional column indicating the assigned cluster for each book. Also includes number of ratings and average rating for each book.
            
    -   "books_by_cluster_ratings.csv" : str (CSV file)
        
        -   CSV file containing lists of books, grouped by their assigned cluster. Must correspond to the clusters in "books_clustered_ratings.csv".
            
    -   std_num_outlier : float
        
        -   The number of standard deviations below the mean similarity that serves as the threshold for outliers.
            
    -   highest_hits_to_keep : int
        
        -   The number of books with the highest number of ratings to keep. Should be very high relative to the number of books left after filtering for outliers.
            
    -   highest_ratings_to_keep : int
        
        -   The number of the most highly rated books (in terms of average rating) to keep. Should be less than "highest_hits_to_keep".
            
    -   keyword_weight : float
        
        -   How much keywords should be weighted relative to genres when measuring similarities. The optimal is somewhere around 0.8.
            
-   **Returns:**
    
    -   "recommendations.csv" : CSV file
        
        -   Popular books that share a k-modes cluster with a book the user has read before, sorted by average rating.
            

**8A. CONTENT-BASED FILTERING:** Train the content-based filtering model.

-   `content_filtering\train_similarity_final.py`
    
-   Reads book data with keyword and genre vectors, calculates a combined cosine similarity matrix based on configurable weights, and exports the matrix and corresponding book identifiers.
    
-   **Parameters:**
    
    -   `books_with_vectors_filepath`: str (CSV file path)
        
        -   Path to the CSV file containing book titles, authors, and keyword/genre vectors as string representations of lists (e.g., `../keyword_processing/books_with_vectors.csv` from Keyword Processing).
            
    -   `genre_weight`: float, optional
        
        -   Weight to assign to genre similarity in the combined similarity calculation. Defaults to 0.5.
            
    -   `keyword_weight`: float, optional
        
        -   Weight to assign to keyword similarity in the combined similarity calculation. Defaults to 0.5.
            
    -   `checkpoint_dir`: str, optional
        
        -   Directory path for saving and loading intermediate similarity matrix checkpoints (`.npy` files). If `None`, checkpointing is skipped. Defaults to `./similarity_checkpoints`.
            
-   **Returns:**
    
    -   `book_similarity_matrix_{genre_weight_normalized}.npy`: NumPy `.npy` file
        
        -   A square NumPy array representing the combined similarity matrix between all books. The filename will include the normalized `genre_weight` (e.g., `book_similarity_matrix_g0-8_k0-2.npy`).
            
    -   `book_similarity_matrix_books_{genre_weight_normalized}.txt`: Text file
        
        -   A text file containing a list of book identifiers (e.g., "Title, Author Name"), corresponding to the order of books in the similarity matrix. The filename will include the normalized `genre_weight`.
            
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
    
-   **Parameters:**
    
    -   `ORIGINAL_H5_FILEPATH`: str
        
        -   Path to the original large HDF5 file (e.g., `book_similarity_matrix.h5` generated by `numpy_to_h5.py`).
            
    -   `H5_DATASET_NAME`: str
        
        -   The name of the dataset within the HDF5 file to be split (e.g., `similarity_matrix`).
            
    -   `CHUNK_SIZE_MB`: int
        
        -   The maximum desired size for each output HDF5 chunk file in megabytes.
            
    -   `OUTPUT_DIR`: str
        
        -   The directory where the chunked HDF5 files will be saved.
            
-   **Returns:**
    
    -   `{OUTPUT_DIR}/{original_filename_prefix}_part_{chunk_index}.h5`: HDF5 (.h5) files
        
        -   Multiple HDF5 files, each representing a chunk of the original dataset. These files are saved in the specified `OUTPUT_DIR`.
            

**8B. COLLABORATIVE FILTERING:** Train the collaborative filtering model.

-   `collab_filtering\cf_final.py`
    
-   Trains a collaborative filtering recommendation system using matrix factorization and exports the trained model.
    
-   **Parameters:**

    - `DATA_PATH` : str 

        - The directory where the ratings file is saved (e.g. `../train_test_split/split_data_combined/`)
    
    -   `TRAIN_RATINGS_FILE`: str (CSV file path)
        
        -   Path to the CSV file containing user ratings (e.g., `train_ratings.csv`). This file should include 'User_id', 'Book_id', and 'Book_Rating'.
            
            
-   **Returns:**
    
            
    -   `X_features_{num_features}_lambda_{lambda_val}.npy`: NumPy `.npy` file (saved in `output_dir`)
        
        -   Learned item features (`X`) from the matrix factorization model.
            
    -   `b_item_bias_{num_features}_lambda_{lambda_val}.npy`: NumPy `.npy` file (saved in `output_dir`)
        
        -   Learned item biases (`b_item`) from the matrix factorization model.
            
    -   `book_list_features_{num_features}_lambda_{lambda_val}.txt`: Text file (saved in `output_dir`)
        
        -   A list of unique book identifiers used in the trained model.
            
    -   `Y_mean_features_{num_features}_lambda_{lambda_val}.npy`: NumPy `.npy` file (saved in `output_dir`)
        
        -   The mean rating for each book used in the model.
            

**8C. COMBINED FILTERING:** Generate book recommendation lists using a combination of content-based filtering and collaborative filtering.

-   `combined_model\cb_get_recs_new.py`
    
-   This script calculates content-based recommendations for a user based on their rated books. It utilizes pre-computed similarity data stored in HDF5 chunks to compute weighted similarity scores for all books.
    
-   **Parameters:**
    
    -   `book_identifiers_txt_filepath`: str (file path)
        
        -   Path to the text file containing the book identifiers (e.g., `../content_filtering/book_similarity_matrix_books_0-8.txt`), corresponding to the HDF5 matrix order.
            
    -   `user_books`: list of str
        
        -   A list of book identifiers (e.g., "Title, Author Name") that the user has rated.
            
    -   `user_ratings`: list of int or float
        
        -   A list of corresponding ratings for `user_books` (e.g., 1-5 scale).
            
    -   (Implicit) `H5_CHUNKS_DIR`: str
        
        -   Directory containing the chunked HDF5 similarity matrices (e.g., `../content_filtering/split_h5_chunks`).
            
-   **Returns:**
    
    -   `pd.DataFrame`:
        
        -   A DataFrame with 'Book Identifier', 'Weighted Similarity Score', 'Rated by User' (boolean), and 'Original Rating' columns. The DataFrame is sorted by 'Weighted Similarity Score' in descending order. Returns an empty DataFrame if no valid rated books are found or an error occurs.

            
-   `combined_model\cf_get_recs.py`
    
-   Prepares user ratings, trains a user-specific Ridge regression model using pre-trained book features and biases, and calculates predicted ratings for all books. The output includes user-rated books with their original ratings and a flag indicating if they were rated.
    
-   **Parameters:**
    
    -   `book_features`: `np.ndarray`
        
        -   The full matrix of all book features.
            
    -   `item_bias`: `np.ndarray`
        
        -   The full item bias vector.
            
    -   `mean_book_ratings`: `np.ndarray`
        
        -   The mean rating for all books.
            
    -   `all_book_identifiers`: `list[str]`
        
        -   The master list of all book identifiers.
            
    -   `total_books`: `int`
        
        -   Total number of books.
            
    -   `user_rated_book_identifiers`: `list[str]`
        
        -   List of book identifiers rated by the user.
            
    -   `user_book_ratings`: `list[float]`
        
        -   List of ratings corresponding to user_rated_book_identifiers.
            
    -   `new_user_regularization_strength`: `float`
        
        -   Regularization parameter for the new user model.
            
-   **Returns:**
    
    -   `pd.DataFrame` or `None`: Pandas DataFrame with 'Book Identifier', 'Predicted Rating', 'Rated by User', and 'Original Rating' columns, sorted by 'Predicted Rating' in descending order. Returns `None` if user model training fails.
        
-   `combined_model\combined_get_recs_final.py`
    
-   Generates combined book recommendations using both content-based similarity and collaborative filtering predictions. It loads necessary models and data, then applies multiple strategies (fractional, filtered by CF, multiplied scores, and hybrid) to generate and display recommendation lists.
    
-   **Parameters:**
    
    -   `user_books`: list[str]
        
        -   A list of book identifiers (e.g., "Title, Author Name") that the user has rated.
            
    -   `user_ratings`: list[float]
        
        -   A list of corresponding ratings (e.g., 1-5 scale) for `user_books`.
            
    -   `genre_weight`: int, optional
        
        -   The weight applied to genre similarity in content-based model file paths. Defaults to 0.8.
            
    -   `num_features`: int, optional
        
        -   Number of latent features for the collaborative filtering model. Defaults to 600.
            
    -   `new_user_regularization_strength`: int, optional
        
        -   Regularization strength for the new user CF model. Defaults to 5.
            
-   **Returns:**
    
    -   `pd.DataFrame` or `None`:
        
        -   The primary return is a DataFrame containing:
            
            -   `'Book Identifier'`: Combined book identifier (e.g., "Title, Author Name").
                
            -   `'CB_Weighted_Similarity_Score'`: Content-based similarity score.
                
            -   `'CF_Predicted_Rating'`: Collaborative filtering predicted rating.
                
            -   `'Rated by User'`: Boolean indicating if the book was rated by the user.
                
            -   `'Original Rating'`: The user's original rating for the book (NaN if not rated).
                
        -   This DataFrame is sorted by `CF_Predicted_Rating` then `CB_Weighted_Similarity_Score` in descending order. Returns `None` if data loading or prediction fails.
            
    -   (Implicit) Printed Recommendation Lists:
        
        -   The script also directly prints various recommendation lists to the console based on different strategies:
            
            -   **Strategy 1: CB Top N Filtered by CF:** Recommends books by first taking the top N from content-based recommendations, then filtering and re-ranking that list based on collaborative filtering scores.
                
            -   **Strategy 2: Scores Multiplied Together:** Recommends books by creating a new combined score by multiplying the content-based similarity score and the collaborative filtering predicted rating.
                
            -   **Strategy 3: Pure CB:** Recommends a specified fraction of the top N from content-based recommendations.
                
            -   **Strategy 4: Hybrid Strategy 2+1:** Recommends books by combining a fraction of pure top N content-based recommendations with the remaining fraction from a list of top content-based books re-ranked by collaborative filtering, prioritizing books appearing in both lists.
                

**9. TESTING AND VALIDATION:** Verify the accuracy of book recommendation lists.

-   `combined_model\combined_recs_testing_final.py`
    
-   This script evaluates the performance of various combined recommendation strategies across multiple users. It loads and preprocesses raw user rating data, splits each user's ratings into training and validation sets, and then generates recommendations for each user based on their _training_ data. It compares these generated recommendations against the "relevant" books (highly rated books) in the user's _validation_ set. The script defines several recommendation strategies (pure content-based, pure collaborative filtering, CB filtered by CF, multiplied scores, hybrid, and random) and organizes the results for subsequent metric calculation. This script imports and utilizes `combined_model/cb_get_recs.py` rather than `combined_model/cb_get_recs_new.py` for its content-based filtering logic.
    
-   **Parameters:**
    
    -   `user_ratings_file_path`: str (CSV file path)
        
        -   Path to the input CSV file containing raw user ratings, which will be processed and split for testing (e.g., `../split_data/test_ratings.csv`). This file is expected to have 'User_id', 'Title', 'Author', and 'Rating' columns.
            
    -   `relevance_threshold`: float, optional
        
        -   The minimum rating (inclusive) for a book to be considered "relevant" in the validation set for evaluation purposes. Defaults to 3.0.
            
    -   `validation_split_ratio`: float, optional
        
        -   The fraction (0.0 to 1.0) of a user's rated books to reserve for the validation set during the split. Defaults to 0.5. For a user to be included in the split, they must have at least two ratings.
            
    -   `genre_weight`: int, optional
        
        -   The weight applied to genre similarity in content-based model file paths. Defaults to 1.
            
    -   `num_features`: int, optional
        
        -   Number of latent features for the collaborative filtering model. Defaults to 400.
            
    -   `new_user_regularization_strength`: int, optional
        
        -   Regularization strength for the new user CF model. Defaults to 15.
            
    -   `output_limit_per_strategy`: int, optional
        
        -   The maximum number of recommendations to generate for each strategy for each user during testing. Defaults to 20.
            
-   **Returns:**
    
    -   `pd.DataFrame`: A DataFrame where each row corresponds to a user that had enough ratings to be split. It contains:
        
        -   `'User_id'`: The ID of the user.
            
        -   `'Book_Identifications_List'`: List of book identifiers used for _training_ the user's model.
            
        -   `'Ratings_List'`: List of ratings corresponding to `Book_Identifications_List` (training ratings).
            
        -   `'Relevant_Books_List'`: List of book identifiers from the validation set that the user rated at or above `relevance_threshold`.
            
        -   `'Original_Validation_Ratings_List'`: List of original ratings from the validation set.
            
        -   `'Rec_CB_Only'`: List of recommended book identifiers from the pure content-based strategy.
            
        -   `'Rec_CF_Only'`: List of recommended book identifiers from the pure collaborative filtering strategy.
            
        -   `'Rec_CB_Filtered_by_CF'`: List of recommended book identifiers from the content-based strategy, re-ranked by CF scores.
            
        -   `'Rec_By_Multiplying_Scores'`: List of recommended book identifiers where CB and CF scores are multiplied.
            
        -   `'Rec_Random'`: List of randomly selected recommended book identifiers (as a baseline).
            
-   `combined_model\testing_final.py`
    
-   This script executes the evaluation pipeline. It handles the loading of necessary models and data, processes user ratings, splits data for testing, generates recommendations using `combined_recs_testing_final.py`, and evaluates the overlap of these recommendations with relevant books from the validation set. It includes checkpointing for resuming interrupted runs and handles various configurations. 
    
-   **Parameters:**
    
    -   `test_config`: dict
        
        -   A dictionary defining the configuration for the test run, including `'num_features'`, `'lambda_val'`, and `'genre_weight'`.
            
    -   (Implicit) `csv_file_path`: str
        
        -   Path to the main CSV file containing test user ratings (e.g., `../train_test_split/split_data_combined/test_ratings.csv`).
            
-   **Returns:**
    
    -   `overlap_evaluation_nf{nf}_rs{rs}_gw{gw}.csv`: CSV file
        
        -   A CSV file containing the overlap evaluation results (precision, recall, etc.) for each recommendation strategy, saved within a configuration-specific subdirectory under `CHECKPOINT_ROOT_DIR`. This file is then used as input by `analyze.py`.
            
    -   (Implicit) Checkpoint Files:
        
        -   `overall_pipeline_progress.pkl`: Pickle file
            
            -   Saves the overall progress of the pipeline, including the last completed step and paths to intermediate dataframes for each configuration.
                
        -   Intermediate Parquet files (`.parquet`):
            
            -   `user_data_processed.parquet`, `training_df.parquet`, `validation_df.parquet`, `all_user_recs.parquet`, `relevant_val_books.parquet` are saved within configuration-specific subdirectories for checkpointing and efficient loading.
                
-   `combined_model\analyze.py`
    
-   This script handles the statistical analysis and visualization of the recommendation overlap results generated by `testing_final.py`. It loads the `overlap_evaluation` CSV files, calculates descriptive statistics (mean, median, skew, etc.) for overall performance and within different bins of "total relevant books" (e.g., low, medium, high), and then generates various plots (bar charts, KDE plots) to help interpret the effectiveness of different recommendation strategies. It is the designated visualization component of the pipeline.
    
-   **Parameters:**
    
    -   `checkpoint_root_dir`: str
        
        -   The root directory where the output directories from `testing_final.py` are located (e.g., "single_run_evaluation_results_loop").
            
    -   `parameters_list`: list of dict
        
        -   A list of dictionaries, where each dictionary specifies the configuration (e.g., `{'num_features': 600, 'lambda_val': 5, 'genre_weight': 0.8}`) for which `overlap_evaluation` CSV files are to be analyzed.
            
    -   `output_file_path`: str, optional
        
        -   The path to a text file where the detailed statistical analysis results will be written. If `None`, results are only printed to the console.
            
-   **Returns:**
    
    -   `output_file_path`: Text file
        
        -   A text file containing the detailed descriptive statistics for each processed overlap evaluation file, including overall and binned results.
            
    -   Plot Files (PNG):
        
        -   Various plots are generated and saved within `plots` subdirectories inside each configuration's output folder:
            
            -   `overall_mean_overlap_nf{nf}_rs{rs}_gw{gw}.png`
                
            -   `binned_mean_overlap_nf{nf}_rs{rs}_gw{gw}.png`
                
            -   `distribution_overlap_rec_{strategy_name}_nf{nf}_rs{rs}_gw{gw}.png`
                
-   **MANUAL TESTING**
    
-   The quality of book recommendation lists can be verified by developers by inputting a list of books with ratings out of five stars and then evaluating the quality of the resulting recommendations.