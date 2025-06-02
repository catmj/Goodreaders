import pandas as pd
import sys
import os

def clean_author_id_for_merge(author_id):
    """
    Cleans and standardizes an author ID for merging.
    - Converts to string, handles None/NaN.
    - Strips whitespace.
    - Removes '.0' suffix if it represents a float integer (e.g., '123.0' -> '123').
    """
    if pd.isna(author_id) or str(author_id).strip() == '':
        return ''
    s_id = str(author_id).strip()
    
    # Check if it's a float string representation of an integer (e.g., "123.0")
    if s_id.endswith('.0') and s_id[:-2].isdigit():
        return s_id[:-2] # Remove .0 suffix
    
    # If it's a pure digit string, return as is
    elif s_id.isdigit():
        return s_id
    
    # For other cases, try converting to numeric then int, then string
    try:
        num_id = pd.to_numeric(s_id, errors='coerce')
        if pd.notna(num_id):
            return str(int(num_id))
        else:
            return '' # If not a valid number after coercion
    except Exception:
        return '' # Fallback for unexpected formats

def add_author_names_to_book_data(books_csv_path, authors_csv_path, output_csv_path):
    """
    Combines a CSV file containing book data (titles, URLs, author IDs)
    with another CSV containing author IDs and author names,
    adding author names to the book data based on matching Author IDs.

    Args:
        books_csv_path (str): Path to the CSV file with book titles, URLs, and author IDs.
                              (e.g., 'your_books_data.csv').
        authors_csv_path (str): Path to the CSV file with author IDs and author names.
                                (e.g., 'your_authors_data.csv').
        output_csv_path (str): Path where the combined CSV will be saved.
    """
    if not os.path.exists(books_csv_path):
        print(f"Error: Books CSV file not found at '{books_csv_path}'. Please ensure it exists.", file=sys.stderr)
        return
    if not os.path.exists(authors_csv_path):
        print(f"Error: Authors CSV file not found at '{authors_csv_path}'. Please ensure it exists.", file=sys.stderr)
        return

    try:
        # Load the CSV with book data
        # Assuming 'Author ID' is the column name for author IDs in this file
        books_df = pd.read_csv(books_csv_path, encoding='utf-8')

        # Load the CSV with author names
        # Assuming 'Author ID' and 'Author Name' are the column names in this file
        authors_df = pd.read_csv(authors_csv_path, encoding='utf-8')

        print(f"Original books_df columns: {books_df.columns.tolist()}", file=sys.stderr)
        print(f"Original authors_df columns: {authors_df.columns.tolist()}", file=sys.stderr)

        # Apply cleaning to the 'Author ID' column in both DataFrames
        # This creates a temporary column for robust merging
        books_df['Author ID_cleaned_for_merge'] = books_df['Author ID'].apply(clean_author_id_for_merge)
        authors_df['Author ID_cleaned_for_merge'] = authors_df['Author ID'].apply(clean_author_id_for_merge)

        # Perform the merge operation based on the cleaned Author IDs
        # We'll do a left merge to keep all entries from the books_df
        # and add the matching author names.
        combined_df = pd.merge(
            books_df,
            authors_df[['Author ID_cleaned_for_merge', 'Author Name']], # Select only necessary columns from authors_df
            left_on='Author ID_cleaned_for_merge', # Use the cleaned ID from books_df
            right_on='Author ID_cleaned_for_merge', # Use the cleaned ID from authors_df
            how='left'
        )

        # Drop the temporary cleaned ID column
        combined_df = combined_df.drop(columns=['Author ID_cleaned_for_merge'])

        # Optionally, convert the original 'Author ID' column back to nullable integer
        # after the merge, if you prefer its numeric representation in the final CSV.
        # First, ensure it's numeric, coercing errors to NaN
        combined_df['Author ID'] = pd.to_numeric(combined_df['Author ID'], errors='coerce')
        # Then convert to nullable integer type
        combined_df['Author ID'] = combined_df['Author ID'].astype(pd.Int64Dtype())


        # Reorder columns to place 'Author Name' next to 'Author ID' for clarity
        cols = combined_df.columns.tolist()
        if 'Author ID' in cols and 'Author Name' in cols:
            author_id_idx = cols.index('Author ID')
            # Remove 'Author Name' from its current position
            cols.remove('Author Name')
            # Insert 'Author Name' right after 'Author ID'
            cols.insert(author_id_idx + 1, 'Author Name')
        
        combined_df = combined_df[cols]


        # Save the combined DataFrame to a new CSV file
        combined_df.to_csv(output_csv_path, index=False, encoding='utf-8')

        print(f"Successfully combined data into '{output_csv_path}'.", file=sys.stderr)
        print(f"Total entries in combined file: {len(combined_df)}", file=sys.stderr)
        print("Final columns in the new CSV:", combined_df.columns.tolist(), file=sys.stderr)

    except FileNotFoundError as e:
        print(f"Error: A file was not found. {e}", file=sys.stderr)
    except pd.errors.EmptyDataError:
        print(f"Error: One of the input CSV files is empty.", file=sys.stderr)
    except KeyError as e:
        print(f"Error: Missing expected column for merge. Please check column names in your CSVs. Error: {e}", file=sys.stderr)
        # Print column names for debugging
        if 'books_df' in locals(): print(f"Books CSV columns: {books_df.columns.tolist()}", file=sys.stderr)
        if 'authors_df' in locals(): print(f"Authors CSV columns: {authors_df.columns.tolist()}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)

# --- How to use the function ---
# Replace these with your actual file names/paths
# This is the CSV with book titles, Goodreads URLs, and Author IDs
books_data_csv = 'goodreads_book_data.csv' # e.g., 'reviews_with_goodreads_urls_title_match.csv'
# This is the CSV with Author IDs and Author Names
authors_data_csv = 'goodreads_author_ids_and_names.csv' # e.g., 'goodreads_author_ids_and_names.csv'
# The name for your final combined CSV file
output_combined_csv = 'final_combined_book_data.csv'

# Call the function to combine the CSVs
add_author_names_to_book_data(books_data_csv, authors_data_csv, output_combined_csv)