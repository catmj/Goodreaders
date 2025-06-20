import pandas as pd
import json # Import the json library
from scipy import stats # Import scipy.stats for mode calculation
import numpy as np # Import numpy for robust handling of scipy.stats.mode results

def _load_and_process_key_clusters(key_clusters_filepath):
    """
    Loads keyword cluster data from a CSV file and creates a mapping from keywords to cluster indices.

    Args:
        key_clusters_filepath (str): The path to the CSV file containing key clusters.

    Returns:
        tuple: A tuple containing:
               - clusters (list of lists): List of keyword clusters.
               - keyword_to_cluster_map (dict): A mapping from individual keywords to their cluster index.
               - num_clusters (int): The total number of clusters found.
    Raises:
        FileNotFoundError: If the key_clusters_filepath is not found.
        Exception: For other errors during file reading.
    """
    print(f"Reading key clusters from: {key_clusters_filepath}")
    clusters = []
    keyword_to_cluster_map = {}
    cluster_index = 0
    try:
        with open(key_clusters_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                # Split by comma and aggressively strip each keyword
                current_cluster = [k.strip().strip('"').strip('\\') for k in stripped_line.split(',') if k.strip()]
                if current_cluster: # Ensure the cluster is not empty after stripping
                    clusters.append(current_cluster)
                    for keyword in current_cluster:
                        clean_keyword = keyword.strip().strip('"').strip('\\')
                        if clean_keyword:
                            keyword_to_cluster_map[clean_keyword] = cluster_index
                    cluster_index += 1
        num_clusters = len(clusters)
        print(f"Key clusters extracted successfully. Found {num_clusters} clusters.")
        return clusters, keyword_to_cluster_map, num_clusters
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: key_clusters.csv not found at {key_clusters_filepath}")
    except Exception as e:
        raise Exception(f"An error occurred while reading key clusters: {e}")

def _load_and_process_book_data(books_filepath):
    """
    Loads book data from a CSV file and preprocesses 'genres' and 'keywords' columns.

    Args:
        books_filepath (str): The path to the CSV file containing book data.

    Returns:
        pd.DataFrame: DataFrame containing processed book data.
    Raises:
        FileNotFoundError: If the books_filepath is not found.
        Exception: For other errors during file reading.
    """
    print(f"Reading book data from: {books_filepath}")
    try:
        df = pd.read_csv(books_filepath)

        if 'genres' in df.columns:
            # Safely evaluate string representation of lists, handling potential NaN or non-string values
            df['genres'] = df['genres'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else [])
        else:
            df['genres'] = [[]] * len(df)

        if 'keywords' in df.columns:
            # Safely split keywords, handling potential NaN values
            df['keywords'] = df['keywords'].apply(lambda x: [k.strip() for k in str(x).split(',') if k.strip()] if pd.notna(x) else [])
        else:
            df['keywords'] = [[]] * len(df)

        print(f"Book data extracted successfully. Found {len(df)} books.")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: books.csv not found at {books_filepath}")
    except Exception as e:
        raise Exception(f"An error occurred while reading book data: {e}")

def _generate_vectors(df, keyword_to_cluster_map, num_clusters):
    """
    Generates keyword vectors and genre vectors for each book in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing book data with 'keywords' and 'genres' columns.
        keyword_to_cluster_map (dict): Mapping from keywords to cluster indices.
        num_clusters (int): Total number of keyword clusters.

    Returns:
        tuple: A tuple containing:
               - keyword_vectors (list of lists): Generated keyword vectors.
               - all_genres (list): Sorted list of all unique genres.
               - genre_vectors (list of lists): Generated genre vectors.
               - genre_to_index_map (dict): Mapping from genres to their indices.
    """
    # Generate keyword vectors
    print("\nGenerating keyword vectors...")
    keyword_vectors = []
    for _, row in df.iterrows():
        vector = [0] * num_clusters
        for keyword in row['keywords']:
            clean_keyword = keyword.strip().strip('"').strip('\\')
            if clean_keyword in keyword_to_cluster_map:
                cluster_idx = keyword_to_cluster_map[clean_keyword]
                vector[cluster_idx] = 1
        keyword_vectors.append(vector)
    print("Keyword vectors generated.")

    # Extract unique genres and generate genre vectors
    print("\nExtracting unique genres and generating genre vectors...")
    all_genres = sorted(list(set([genre for sublist in df['genres'] for genre in sublist])))
    genre_to_index_map = {genre: i for i, genre in enumerate(all_genres)}
    num_genres = len(all_genres)
    print(f"Found {num_genres} unique genres.")

    genre_vectors = []
    for _, row in df.iterrows():
        vector = [0] * num_genres
        for genre in row['genres']:
            if genre in genre_to_index_map:
                genre_idx = genre_to_index_map[genre]
                vector[genre_idx] = 1
        genre_vectors.append(vector)
    print("Genre vectors generated.")

    return keyword_vectors, all_genres, genre_vectors, genre_to_index_map


def process_book_data_from_csv(key_clusters_filepath, books_filepath):
    """
    Orchestrates the loading, preprocessing, and vector generation for book data.

    Args:
        key_clusters_filepath (str): Path to the CSV file containing key clusters.
        books_filepath (str): Path to the CSV file containing book data.

    Returns:
        tuple: A tuple containing:
               - df (pd.DataFrame): Processed DataFrame with lists for genres and keywords.
               - keyword_vectors (list of lists): Vectors where each represents if a book has a keyword in a certain cluster.
               - all_genres (list): Sorted list of all unique genres found.
               - genre_vectors (list of lists): Vectors where each represents if a book belongs to certain genres.
               - keyword_to_cluster_map (dict): Mapping from individual keywords to their cluster index.
               - genre_to_index_map (dict): Mapping from unique genres to their index.
    """
    try:
        clusters, keyword_to_cluster_map, num_clusters = _load_and_process_key_clusters(key_clusters_filepath)
        if num_clusters == 0:
            print("Warning: No key clusters were extracted. Keyword vectors will be empty.")

        df = _load_and_process_book_data(books_filepath)

        keyword_vectors, all_genres, genre_vectors, genre_to_index_map = \
            _generate_vectors(df, keyword_to_cluster_map, num_clusters)

        return df, keyword_vectors, all_genres, genre_vectors, keyword_to_cluster_map, genre_to_index_map

    except FileNotFoundError as e:
        print(e)
        return pd.DataFrame(), [], [], [], {}, {}
    except Exception as e:
        print(f"An unexpected error occurred during data processing: {e}")
        return pd.DataFrame(), [], [], [], {}, {}


def export_vectors_to_csv(books_df, keyword_vecs, genre_vecs, output_filename='books_with_vectors.csv'):
    """
    Combines book title, author, keyword vectors, genre vectors, and their sums
    into a DataFrame, sorts it by the sums, and exports it to a CSV file.
    Also prints the average, median, range, min, max, mode, standard deviation, and skewness of the sum columns.

    Args:
        books_df (pd.DataFrame): The original DataFrame containing book titles and authors.
        keyword_vecs (list of lists): The generated keyword vectors.
        genre_vecs (list of lists): The generated genre vectors.
        output_filename (str): The name of the CSV file to export.
    """
    if books_df.empty:
        print("No book data to export.")
        return

    print(f"\nPreparing data for export to {output_filename}...")

    # Select relevant columns from the original book DataFrame
    export_df = books_df[['title', 'author_name']].copy()

    # Add keyword vectors as a single column (list representation)
    export_df['keyword_vector'] = keyword_vecs
    # Calculate sum of keyword vector and add as a new column
    export_df['keyword_vector_sum'] = export_df['keyword_vector'].apply(sum)

    # Add genre vectors as a single column (list representation)
    export_df['genre_vector'] = genre_vecs
    # Calculate sum of genre vector and add as a new column
    export_df['genre_vector_sum'] = export_df['genre_vector'].apply(sum)

    # Calculate and print statistics for keyword_vector_sum
    if not export_df['keyword_vector_sum'].empty:
        kw_avg = export_df['keyword_vector_sum'].mean()
        kw_median = export_df['keyword_vector_sum'].median()
        kw_min = export_df['keyword_vector_sum'].min()
        kw_max = export_df['keyword_vector_sum'].max()
        kw_range = kw_max - kw_min
        kw_std = export_df['keyword_vector_sum'].std() # Standard deviation
        kw_skewness = export_df['keyword_vector_sum'].skew() # Skewness
        
        kw_mode_result = stats.mode(export_df['keyword_vector_sum'])
        # Safely extract the mode using np.atleast_1d to handle cases where mode might be a scalar
        # or an empty array from scipy.stats.mode.
        kw_mode_array = np.atleast_1d(kw_mode_result.mode)
        kw_mode = kw_mode_array[0] if kw_mode_array.size > 0 else "N/A"

        print(f"\nKeyword Vector Sum Statistics:")
        print(f"  Average: {kw_avg:.2f}")
        print(f"  Median: {kw_median:.2f}")
        print(f"  Min: {kw_min:.2f}")
        print(f"  Max: {kw_max:.2f}")
        print(f"  Range: {kw_range:.2f}")
        print(f"  Mode: {kw_mode}")
        print(f"  Standard Deviation: {kw_std:.2f}")
        print(f"  Skewness: {kw_skewness:.2f}") # Print skewness
    else:
        print("\nNo keyword vector sums to calculate statistics for.")

    # Calculate and print statistics for genre_vector_sum
    if not export_df['genre_vector_sum'].empty:
        genre_avg = export_df['genre_vector_sum'].mean()
        genre_median = export_df['genre_vector_sum'].median()
        genre_min = export_df['genre_vector_sum'].min()
        genre_max = export_df['genre_vector_sum'].max()
        genre_range = genre_max - genre_min
        genre_std = export_df['genre_vector_sum'].std() # Standard deviation
        genre_skewness = export_df['genre_vector_sum'].skew() # Skewness
        
        genre_mode_result = stats.mode(export_df['genre_vector_sum'])
        # Safely extract the mode using np.atleast_1d for robustness
        genre_mode_array = np.atleast_1d(genre_mode_result.mode)
        genre_mode = genre_mode_array[0] if genre_mode_array.size > 0 else "N/A"

        print(f"\nGenre Vector Sum Statistics:")
        print(f"  Average: {genre_avg:.2f}")
        print(f"  Median: {genre_median:.2f}")
        print(f"  Min: {genre_min:.2f}")
        print(f"  Max: {genre_max:.2f}")
        print(f"  Range: {genre_range:.2f}")
        print(f"  Mode: {genre_mode}")
        print(f"  Standard Deviation: {genre_std:.2f}")
        print(f"  Skewness: {genre_skewness:.2f}") # Print skewness
    else:
        print("\nNo genre vector sums to calculate statistics for.")

    # Sort the DataFrame by the sum columns (descending order)
    # You can change ascending=False to True if you want to sort in ascending order
    export_df.sort_values(by=['keyword_vector_sum', 'genre_vector_sum'], ascending=[False, False], inplace=True)


    # Export to CSV
    try:
        # Convert lists to string representation for CSV if desired, otherwise they might appear as '[1, 0, 1]'
        # Note: eval() would be needed to convert back to list when reading this CSV
        export_df['keyword_vector'] = export_df['keyword_vector'].apply(str)
        export_df['genre_vector'] = export_df['genre_vector'].apply(str)

        export_df.to_csv(output_filename, index=False, encoding='utf-8')
        print(f"Successfully exported data to '{output_filename}'")
    except Exception as e:
        print(f"Error exporting data to CSV: {e}")

def export_dictionaries_to_json(dictionary, output_filename):
    """
    Exports a dictionary to a JSON file.

    Args:
        dictionary (dict): The dictionary to export.
        output_filename (str): The name of the JSON file to create.
    """
    print(f"Exporting dictionary to '{output_filename}'...")
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(dictionary, f, indent=4, ensure_ascii=False)
        print(f"Successfully exported dictionary to '{output_filename}'")
    except Exception as e:
        print(f"Error exporting dictionary to JSON: {e}")




# --- Define your CSV file paths and output JSON file names ---
key_clusters_file = "../keyword_clustering/keys_by_cluster.csv"
books_file = "../keyword_analysis_keyBERT/output_file_books_train_preprocessed_n15_div0.2.csv"
output_csv_file = 'books_with_vectors.csv'
keyword_map_json_file = 'keyword_to_cluster_map.json'
genre_map_json_file = 'genre_to_index_map.json'


# --- Call the main processing function ---
books_df_processed, keyword_vectors_output, unique_genres_list, genre_vectors_output, \
    keyword_map_output, genre_map_output = process_book_data_from_csv(key_clusters_file, books_file)

# --- Display Results (Optional - for verification) ---
print("\n--- Results ---")

if not books_df_processed.empty:
    print("\nProcessed Book Data (first 2 rows):")
    print(books_df_processed.head(2))

    print("\nKeyword Vectors (first 2 books, columns represent clusters):")
    num_clusters_output = len(keyword_vectors_output[0]) if keyword_vectors_output and keyword_vectors_output[0] else 0
    if num_clusters_output > 0:
        print(pd.DataFrame(keyword_vectors_output, columns=[f'Cluster_{i}' for i in range(num_clusters_output)]).head(2))
    else:
        print("No keyword vectors generated (perhaps no clusters or books found).")

    print("\nUnique Genres:")
    print(unique_genres_list)

    print("\nGenre Vectors (first 2 books, columns represent genres):")
    if unique_genres_list:
        print(pd.DataFrame(genre_vectors_output, columns=unique_genres_list).head(2))
    else:
        print("No genre vectors generated (perhaps no unique genres found).")
else:
    print("\nNo data was processed successfully.")

# --- Export the combined data to CSV ---
export_vectors_to_csv(books_df_processed, keyword_vectors_output, genre_vectors_output, output_csv_file)

# --- Export the dictionaries to JSON ---
if keyword_map_output:
    export_dictionaries_to_json(keyword_map_output, keyword_map_json_file)
if genre_map_output:
    export_dictionaries_to_json(genre_map_output, genre_map_json_file)
