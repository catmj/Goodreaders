import pandas as pd
import ast
import os

# --- Configuration ---
DATA_PATH = '../train_test_split/split_data_combined/'
RATINGS_FILE = DATA_PATH + 'books_train.csv'


####


def export_books_for_narrowed_recommendations(books_df, output_csv_path):
    """
    Exports a list of book titles and authors from a DataFrame into a CSV file
    with a single 'book' column containing 'title, author' strings,
    compatible with load_narrowed_recommendations.

    Args:
        books_df (pandas.DataFrame): DataFrame containing book data with 'title' and 'author_name' columns.
        output_csv_path (str): The path to the output CSV file.
    """
    if books_df.empty:
        print("No books to export for narrowed recommendations.")
        return

    # Ensure the directory exists
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Combine 'title' and 'author_name' into a single 'book' column
    # Handle potential NaN values by converting to string before stripping
    books_df['combined_book_title'] = books_df.apply(
        lambda row: f"{str(row['title']).strip()}, {str(row['author_name']).strip()}",
        axis=1
    )

    # Create a new DataFrame with just the 'book' column
    export_df = pd.DataFrame({'book': books_df['combined_book_title']})

    # Export to CSV without the index
    export_df.to_csv(output_csv_path, index=False)
    print(f"Exported {len(books_df)} books to '{output_csv_path}' for narrowed recommendations.")

# --- Main Execution ---
if __name__ == "__main__":
    file_name = RATINGS_FILE
    
    # New output file for narrowed recommendations (CSV format)
    narrowed_recommendations_output_csv = 'narrowed_recommendations.csv'
    
    # Define the genres you're interested in
    desired_genres = [
        'Fantasy', 'Science Fiction', 'M M Romance', 'High Fantasy',
        'LGBTQ+', 'Queer', 'Gay', 'Lesbian', 'Transgender', 'Nonbinary', 
        'Speculative Fiction', 'Urban Fantasy', 'Dark Fantasy', 'Epic Fantasy', 
        'Space Opera', 'Dystopian', 'Cyberpunk', 
        'Romance', 'Paranormal Romance', 'Fantasy Romance', 
        'Magical Realism', 
        'Young Adult Fantasy', 'Young Adult Science Fiction', 'New Adult' 
    ]


    def get_books_by_genres(file_path, target_genres):
        """
        Reads a CSV file, parses the 'genres' column, and filters books
        that contain any of the specified target genres (case-insensitive).
        """
        try:
            df = pd.read_csv(file_path)
            # --- ADD THESE LINES TO INSPECT ---
            print("\n--- DataFrame Columns ---")
            print(df.columns)
            print("-------------------------\n")
            # --- END OF INSPECTION LINES ---

        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
            return pd.DataFrame()
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return pd.DataFrame()

        target_genres_lower = {genre.lower() for genre in target_genres}
        df['parsed_genres'] = df['genres'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )
        filtered_books = df[
            df['parsed_genres'].apply(
                lambda book_genres: any(
                    g.lower() in target_genres_lower for g in book_genres
                )
            )
        ].copy()
        filtered_books = filtered_books.drop(columns=['parsed_genres'])
        return filtered_books


    matching_books_df = get_books_by_genres(file_name, desired_genres)

    if not matching_books_df.empty:
        print(f"Found {len(matching_books_df)} books matching the desired genres:")
        # Display relevant columns for verification
        print(matching_books_df[['title', 'author_name', 'genres', 'rating', 'num_ratings']].head(10))
        
        # This function now correctly exports 'title, author' to the 'book' column
        export_books_for_narrowed_recommendations(matching_books_df, narrowed_recommendations_output_csv)

    else:
        print("No books found matching the specified genre criteria.")
