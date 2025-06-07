import pandas as pd
import numpy as np
import os
from pathlib import Path # For better path handling

# --- Configuration ---
# Define base paths for input and output directories
INPUT_DIR = Path('../files_to_clean')
OUTPUT_DIR = Path('../cleaned_data_2')
MAX_FILE_SIZE_MB = 45 # Maximum size for output CSV chunks

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Data Loading ---
# Load primary datasets
try:
    books = pd.read_csv(INPUT_DIR / 'books_full_unmatched_uncleaned_1.csv')
    reviews = pd.read_csv(INPUT_DIR / 'books_by_user.csv')
except FileNotFoundError as e:
    print(f"Error loading initial CSV files: {e}. Please ensure 'books_full_unmatched_uncleaned_1.csv' and 'books_by_user.csv' are in {INPUT_DIR}")
    exit() # Exit if essential files are missing

# List of additional review files to load and concatenate
additional_review_files = [
    'books_by_user2.csv',
    'books_by_user23.csv',
    'books_by_user24.csv',
]

# List of additional book files to load and concatenate
additional_book_files = [
    'books_full_unmatched_uncleaned_2.csv',
    'books_full_unmatched_uncleaned_3.csv'
]

# Define the expected columns for review files
REVIEW_COLUMNS = [
    'Title', 'Author', 'Rating', 'User_id', 'isbn', 'num_ratings',
    'avg_rating', 'num_pages', 'review', 'date_started', 'date_read',
    'date_added', 'format', 'date_pub_edition'
]

# Define the expected columns for book files
BOOK_COLUMNS = [
    "index","category","genres","votes","title","description","author_name",
    "author_about","avatar_url","pages","rating","num_ratings","num_reviews",
    "publication_info","img_url","book_url","ratings_hist","want_to_read_num","current_read_num"
]

# Load and concatenate additional review files
loaded_additional_reviews = []
for file_name in additional_review_files:
    file_path = INPUT_DIR / file_name
    if file_path.exists():
        try:
            df = pd.read_csv(file_path)
            # Assign columns if they are not already set (or if they are generic integer columns)
            # This handles cases where headers might be missing or incorrect
            if list(df.columns) != REVIEW_COLUMNS:
                if len(df.columns) == len(REVIEW_COLUMNS):
                    df.columns = REVIEW_COLUMNS
                else:
                    print(f"Warning: {file_name} has {len(df.columns)} columns, expected {len(REVIEW_COLUMNS)}. Skipping this file.")
                    continue
            loaded_additional_reviews.append(df)
            print(f"Successfully loaded {file_name}")
        except pd.errors.EmptyDataError:
            print(f"Warning: {file_name} is empty. Skipping.")
        except Exception as e:
            print(f"Error loading {file_name}: {e}. Skipping.")
    else:
        print(f"Info: {file_name} not found at {file_path}. Skipping.")

if loaded_additional_reviews:
    reviews = pd.concat([reviews] + loaded_additional_reviews, ignore_index=True)
    print(f"Total reviews after concatenation: {len(reviews)} rows.")
else:
    print("No additional review files were loaded or concatenated.")


# Load and concatenate additional review files
loaded_additional_books = []
for file_name in additional_book_files:
    file_path = INPUT_DIR / file_name
    if file_path.exists():
        try:
            df = pd.read_csv(file_path)
            # Assign columns if they are not already set (or if they are generic integer columns)
            # This handles cases where headers might be missing or incorrect
            if list(df.columns) != BOOK_COLUMNS:
                if len(df.columns) == len(BOOK_COLUMNS):
                    df.columns = BOOK_COLUMNS
                else:
                    print(f"Warning: {file_name} has {len(df.columns)} columns, expected {len(BOOK_COLUMNS)}. Skipping this file.")
                    continue
            loaded_additional_books.append(df)
            print(f"Successfully loaded {file_name}")
        except pd.errors.EmptyDataError:
            print(f"Warning: {file_name} is empty. Skipping.")
        except Exception as e:
            print(f"Error loading {file_name}: {e}. Skipping.")
    else:
        print(f"Info: {file_name} not found at {file_path}. Skipping.")

if loaded_additional_books:
    books = pd.concat([books] + loaded_additional_books, ignore_index=True)
    print(f"Total books after concatenation: {len(books)} rows.")
else:
    print("No additional review files were loaded or concatenated.")


# --- Date Conversion ---
print("Converting date columns...")
date_columns_books = ["publication_info"]
for col in date_columns_books:
    if col in books.columns:
        books[col] = pd.to_datetime(books[col], errors="coerce")
    else:
        print(f"Warning: '{col}' not found in 'books' DataFrame.")

date_columns_reviews = ["date_started", "date_added", "date_read", "date_pub_edition"]
for col in date_columns_reviews:
    if col in reviews.columns:
        reviews[col] = pd.to_datetime(reviews[col], errors="coerce")
    else:
        print(f"Warning: '{col}' not found in 'reviews' DataFrame.")

print("Date conversion complete.")


# --- Cleaning Functions ---
def clean_author(author_name: str) -> str:
    """
    Cleans author names by removing '*' and reordering "Last, First" to "First Last".
    Example: "Doe, John*" -> "John Doe"
    """
    if not isinstance(author_name, str):
        return str(author_name) # Handle non-string types gracefully
    
    pieces = author_name.replace("*", "").split(", ")
    if len(pieces) == 2:
        return f"{pieces[1]} {pieces[0]}"
    return author_name

def clean_title(title: str) -> str:
    """
    Cleans review titles by taking the first part before a newline or colon.
    Handles a specific "Black Panther" case.
    """
    if not isinstance(title, str):
        return str(title) # Handle non-string types gracefully

    # Specific override for a known problematic title
    if title.startswith("Black Panther, Vol. 1"): # Use startswith for robustness
        return "black panther (2016-2018) #1"
    
    # Split by newline first, then by colon
    first_split = title.split("\n")[0]
    return first_split.split(":")[0]

def clean_title_2(title: str) -> str:
    """
    Cleans book titles based on a predefined list of problematic titles
    that end with '...' or have specific numeric suffixes.
    """
    if not isinstance(title, str):
        return str(title) # Handle non-string types gracefully

    # Using a dictionary for cleaner mapping of specific title corrections
    title_corrections = {
        "The Terrible and Wonderful Reasons Why I Run Long Distances (...": 
            "The Terrible and Wonderful Reasons Why I Run Long Distances (Volume 5)",
        "Death Weavers (4)": "Death Weavers",
        "The Girl Who Circumnavigated Fairyland in a Ship of Her Own M...": 
            "the girl who circumnavigated fairyland in a ship of her own making",
        "But What If We're Wrong? Thinking About the Present As If It ...":
            "but what if we're wrong? thinking about the present as if it were the past",
        "5 Very Good Reasons to Punch a Dolphin in the Mouth and Other...":
            "5 very good reasons to punch a dolphin in the mouth and other useful guides"
    }
    return title_corrections.get(title, title) # Return corrected title or original if not found



# --- Apply Cleaning Functions and Normalize Text ---
print("Applying cleaning functions and normalizing text...")
# Convert to string first to avoid errors with non-string types, then apply map
reviews["Author"] = reviews["Author"].astype(str).map(clean_author).str.lower()
reviews["Title"] = reviews["Title"].astype(str).map(clean_title).str.lower()
books["author_name"] = books["author_name"].astype(str).str.lower()
books["title"] = books["title"].astype(str).map(clean_title_2).str.lower()
print("Cleaning and normalization complete.")



# --- Data Matching ---
print("Matching books and reviews...")
# Create sets of (title, author) pairs for efficient lookup
# Using .dropna() before creating sets to avoid NaN issues
pairs_books = set(tuple(x) for x in books[['title', 'author_name']].dropna().values)
pairs_reviews = set(tuple(x) for x in reviews[['Title', 'Author']].dropna().values)

# Find common pairs
common_pairs = pairs_books.intersection(pairs_reviews)
print(f"Found {len(common_pairs)} common book-author pairs.")

# Filter DataFrames based on common pairs
# Using .isin() for potentially better performance than apply with lambda for large DFs
books_matched = books[books.apply(lambda row: (row['title'], row['author_name']) in common_pairs, axis=1)]
reviews_matched = reviews[reviews.apply(lambda row: (row['Title'], row['Author']) in common_pairs, axis=1)]
reviews_matched = reviews_matched.sample(frac=1)

books_unmatched = books[~books.apply(lambda row: (row['title'], row['author_name']) in common_pairs, axis=1)]
reviews_unmatched = reviews[~reviews.apply(lambda row: (row['Title'], row['Author']) in common_pairs, axis=1)]

# For reviews_unmatched, keep only unique title-author pairs and sort
reviews_unmatched_unique = reviews_unmatched[["Title", "Author"]].drop_duplicates().sort_values(by=['Title', 'Author'])
print("Data matching complete.")


# --- Save Cleaned Data ---
print("Saving cleaned data to CSV files...")
# Save matched and unmatched dataframes
books_matched.to_csv(OUTPUT_DIR / 'books_matched.csv', index=False)
reviews_matched.to_csv(OUTPUT_DIR / 'reviews_matched.csv', index=False)
books_unmatched.to_csv(OUTPUT_DIR / 'books_unmatched.csv', index=False)
reviews_unmatched_unique.to_csv(OUTPUT_DIR / 'reviews_unmatched.csv', index=False) # Use the unique unmatched reviews

print("Standard data saving complete.")


# --- Save matcheded files into smaller file chunks ---


def split_into_chunks(frame_matched: pd.DataFrame, OUTPUT_DIR: Path, MAX_FILE_SIZE_MB: int, file_name_prefix: str):
    """
    Splits a DataFrame into multiple CSV files, each with a maximum specified size.

    Args:
        frame_matched (pd.DataFrame): The DataFrame containing the data to be split.
        OUTPUT_DIR (Path): The directory where the chunked CSV files will be saved.
        MAX_FILE_SIZE_MB (int): The maximum desired size for each output file in megabytes.
        file_name_prefix (str): The prefix for the output file names (e.g., "reviews_matched", "product_data").
    """
    print(f"Splitting '{file_name_prefix}.csv' into chunks (max {MAX_FILE_SIZE_MB}MB per file)...")
    max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024

    total_rows = len(frame_matched)
    if total_rows == 0:
        print("No data to split.")
        return

    # Estimate average row size to calculate initial chunk size
    temp_sample_file = OUTPUT_DIR / '__temp_sample.csv'
    
    # Ensure the output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save a small sample to a temp file to get an accurate row size estimate
    sample_rows = min(100, total_rows)
    if sample_rows > 0:
        frame_matched.head(sample_rows).to_csv(temp_sample_file, index=False)
        sample_size_bytes = os.path.getsize(temp_sample_file)
        os.remove(temp_sample_file)
    else:
        sample_size_bytes = 0

    avg_row_size_bytes = sample_size_bytes / sample_rows if sample_rows > 0 and sample_size_bytes > 0 else 100
    
    # Calculate target rows per chunk
    target_rows_per_chunk = int(max_size_bytes / avg_row_size_bytes)
    if target_rows_per_chunk == 0:
        target_rows_per_chunk = 1 

    current_start_row = 0
    file_index = 1

    while current_start_row < total_rows:
        current_end_row = min(current_start_row + target_rows_per_chunk, total_rows)
        chunk = frame_matched.iloc[current_start_row:current_end_row]
        
        chunk_file_path = OUTPUT_DIR / f'{file_name_prefix}_part_{file_index}.csv'
        chunk.to_csv(chunk_file_path, index=False)
        
        actual_size = os.path.getsize(chunk_file_path)
        print(f"Saved {chunk_file_path} with {len(chunk)} rows (Size: {actual_size / (1024 * 1024):.2f} MB)")
        
        current_start_row = current_end_row
        file_index += 1

    print("All processing and saving complete.")


split_into_chunks(reviews_matched, OUTPUT_DIR, MAX_FILE_SIZE_MB, "reviews_matched")
split_into_chunks(books_matched, OUTPUT_DIR, MAX_FILE_SIZE_MB, "books_matched")