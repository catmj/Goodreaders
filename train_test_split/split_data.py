import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- Configuration ---
DATA_PATH = '../cleaned_data_2/'
OUTPUT_PATH = Path('split_data/')
RATINGS_FILE = DATA_PATH + 'reviews_matched.csv'
BOOK_FILE = DATA_PATH + 'books_matched.csv'

# Define maximum file size for chunking
MAX_FILE_SIZE_MB = 45

# Ensure the output directory exists
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# --- Load Data ---
ratings_full = pd.read_csv(RATINGS_FILE)
books_full = pd.read_csv(BOOK_FILE)

print(f"Loaded full ratings data: {len(ratings_full)} entries")
print(f"Loaded full books data: {len(books_full)} entries")

# --- Remove Duplicates from books_full ---
initial_books_count = len(books_full)
books_full.drop_duplicates(subset=['title', 'author_name'], inplace=True)
print(f"Removed {initial_books_count - len(books_full)} duplicate book entries (title, author_name) from books_full.")
print(f"Books data after removing duplicates: {len(books_full)} entries")

# --- Split Review Data by User (from full ratings data) ---

# Count reviews per user
user_counts = ratings_full["User_id"].value_counts().reset_index()
user_counts.columns = ['User_id', 'Count']

# Identify users with at least 8 reviews
testable_users = user_counts.loc[user_counts['Count'] > 7, "User_id"]

print("\n--- User Split Statistics ---")
print(f"Testable users (with > 7 reviews): {len(testable_users)}")

# Splitting Testable Users for training, testing, and cross-validation
testable_train_users = testable_users.sample(frac=0.6, random_state=13)
print(f"Testable train users (subset of testable users for training): {len(testable_train_users)}")

test_and_cross_val_users = testable_users[~testable_users.isin(testable_train_users)]
print(f"Test and Cross-validation users: {len(test_and_cross_val_users)}")

# All users for the main training set (including those with <= 10 reviews)
train_users = user_counts.loc[~user_counts['User_id'].isin(test_and_cross_val_users), 'User_id']
print(f"Overall train users: {len(train_users)}")

# Sample 50% of test_and_cross_val for cross-validation
cross_val_users = test_and_cross_val_users.sample(frac=0.5, random_state=13)
print(f"Cross-validation users: {len(cross_val_users)}")

# The remaining users from test_and_cross_val go into the test set
test_users = test_and_cross_val_users[~test_and_cross_val_users.isin(cross_val_users)]
print(f"Test users: {len(test_users)}")

# --- Initial Filtering of Ratings Data based on User Splits ---
train_ratings = ratings_full[ratings_full['User_id'].isin(train_users)].copy()
cross_val_ratings = ratings_full[ratings_full['User_id'].isin(cross_val_users)].copy()
test_ratings = ratings_full[ratings_full['User_id'].isin(test_users)].copy()

print(f"\nInitial train ratings count: {len(train_ratings)}")
print(f"Initial cross_val ratings count: {len(cross_val_ratings)}")
print(f"Initial test ratings count: {len(test_ratings)}")

# --- Initial Books Train-Test Split ---
books_train, books_test = train_test_split(books_full, test_size=0.2, random_state=13)

print(f"\nInitial books_train count: {len(books_train)}")
print(f"Initial books_test count: {len(books_test)}")

# --- Matching Books Across Training and Cross-Validation Sets ---
print("\n--- Matching books across train_ratings, cross_val_ratings, and books_train ---")

# 1. Get unique (title, author) pairs from each dataset
train_ratings_book_pairs = set(tuple(x) for x in train_ratings[['Title', 'Author']].dropna().values)
cross_val_ratings_book_pairs = set(tuple(x) for x in cross_val_ratings[['Title', 'Author']].dropna().values)
books_train_book_pairs = set(tuple(x) for x in books_train[['title', 'author_name']].dropna().values) # Corrected as per your confirmation

# 2. Find the intersection of all three sets of book pairs
common_training_books = train_ratings_book_pairs.intersection(
    cross_val_ratings_book_pairs,
    books_train_book_pairs
)
print(f"Found {len(common_training_books)} common book-author pairs for training/cross-validation.")

# 3. Filter the datasets based on these common books
def filter_by_common_books(df, title_col, author_col, common_pairs):
    temp_df = df.copy()
    temp_df['__temp_pair__'] = list(zip(temp_df[title_col], temp_df[author_col]))
    filtered_df = temp_df[temp_df['__temp_pair__'].isin(common_pairs)].drop(columns=['__temp_pair__'])
    return filtered_df

# Apply filtering
train_ratings_matched = filter_by_common_books(train_ratings, 'Title', 'Author', common_training_books)
cross_val_ratings_matched = filter_by_common_books(cross_val_ratings, 'Title', 'Author', common_training_books)
books_train_matched = filter_by_common_books(books_train, 'title', 'author_name', common_training_books) # Corrected as per your confirmation

print(f"Filtered train ratings count: {len(train_ratings_matched)}")
print(f"Filtered cross_val ratings count: {len(cross_val_ratings_matched)}")
print(f"Filtered books_train count: {len(books_train_matched)}")

# --- Function to Split DataFrames into Chunks ---
def split_into_chunks(frame_to_split: pd.DataFrame, output_dir: Path, max_file_size_mb: int, file_name_prefix: str):
    """
    Splits a DataFrame into multiple CSV files, each with a maximum specified size.

    Args:
        frame_to_split (pd.DataFrame): The DataFrame containing the data to be split.
        output_dir (Path): The directory where the chunked CSV files will be saved.
        max_file_size_mb (int): The maximum desired size for each output file in megabytes.
        file_name_prefix (str): The prefix for the output file names (e.g., "reviews_matched", "product_data").
    """
    print(f"\nSplitting '{file_name_prefix}.csv' into chunks (max {max_file_size_mb}MB per file)...")
    max_size_bytes = max_file_size_mb * 1024 * 1024

    total_rows = len(frame_to_split)
    if total_rows == 0:
        print(f"No data in '{file_name_prefix}' to split.")
        return

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Estimate average row size to calculate initial chunk size
    temp_sample_file = output_dir / '__temp_sample.csv'

    # Save a small sample to a temp file to get an accurate row size estimate
    sample_rows = min(100, total_rows)
    if sample_rows > 0:
        frame_to_split.head(sample_rows).to_csv(temp_sample_file, index=False)
        sample_size_bytes = os.path.getsize(temp_sample_file)
        os.remove(temp_sample_file)
    else:
        sample_size_bytes = 0

    avg_row_size_bytes = sample_size_bytes / sample_rows if sample_rows > 0 else 1000
    if avg_row_size_bytes == 0:
        avg_row_size_bytes = 1000

    target_rows_per_chunk = int(max_size_bytes / avg_row_size_bytes)
    if target_rows_per_chunk == 0:
        target_rows_per_chunk = 1

    current_start_row = 0
    file_index = 1

    while current_start_row < total_rows:
        current_end_row = min(current_start_row + target_rows_per_chunk, total_rows)
        chunk = frame_to_split.iloc[current_start_row:current_end_row]

        chunk_file_path = output_dir / f'{file_name_prefix}_part_{file_index}.csv'
        chunk.to_csv(chunk_file_path, index=False)

        actual_size = os.path.getsize(chunk_file_path)
        print(f"  Saved {chunk_file_path.name} with {len(chunk)} rows (Size: {actual_size / (1024 * 1024):.2f} MB)")

        current_start_row = current_end_row
        file_index += 1

    print(f"Finished splitting '{file_name_prefix}'.")

# --- Saving and Chunking All Data ---
print("\n--- Saving and Chunking All Data ---")

# --- train_ratings ---
file_prefix_train_ratings = 'train_ratings'
# Save unchunked version
train_ratings_matched.to_csv(OUTPUT_PATH / f'{file_prefix_train_ratings}.csv', index=False)
print(f"Saved unchunked {file_prefix_train_ratings}.csv")
# Save chunked version
split_into_chunks(train_ratings_matched, OUTPUT_PATH, MAX_FILE_SIZE_MB, file_prefix_train_ratings)

# --- cross_val_ratings ---
file_prefix_cross_val_ratings = 'cross_val_ratings'
# Save unchunked version
cross_val_ratings_matched.to_csv(OUTPUT_PATH / f'{file_prefix_cross_val_ratings}.csv', index=False)
print(f"Saved unchunked {file_prefix_cross_val_ratings}.csv")
# Save chunked version
split_into_chunks(cross_val_ratings_matched, OUTPUT_PATH, MAX_FILE_SIZE_MB, file_prefix_cross_val_ratings)

# --- test_ratings ---
file_prefix_test_ratings = 'test_ratings'
# Save unchunked version
test_ratings.to_csv(OUTPUT_PATH / f'{file_prefix_test_ratings}.csv', index=False)
print(f"Saved unchunked {file_prefix_test_ratings}.csv")
# Save chunked version
split_into_chunks(test_ratings, OUTPUT_PATH, MAX_FILE_SIZE_MB, file_prefix_test_ratings)

# --- books_train ---
file_prefix_books_train = 'books_train'
# Save unchunked version
books_train_matched.to_csv(OUTPUT_PATH / f'{file_prefix_books_train}.csv', index=False)
print(f"Saved unchunked {file_prefix_books_train}.csv")
# Save chunked version
split_into_chunks(books_train_matched, OUTPUT_PATH, MAX_FILE_SIZE_MB, file_prefix_books_train)

# --- books_test ---
file_prefix_books_test = 'books_test'
# Save unchunked version
books_test.to_csv(OUTPUT_PATH / f'{file_prefix_books_test}.csv', index=False)
print(f"Saved unchunked {file_prefix_books_test}.csv")
# Save chunked version
split_into_chunks(books_test, OUTPUT_PATH, MAX_FILE_SIZE_MB, file_prefix_books_test)

print("\nAll data splits, matching, and chunking completed successfully!")