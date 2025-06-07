# Save books matched into smaller file chunks to fit on Github.
import os
import pandas as pd

# Parameters
books_full_unmatched_uncleaned = pd.read_csv('books_full_unmatched_uncleaned.csv')
max_size_mb = 95
max_size_bytes = max_size_mb * 1024 * 1024
base_filename = '../books_full_unmatched_uncleaned'

# Loop to split and save.
start = 0
file_index = 1

# Guess chunk size (start with a few thousand rows, then adjust).
chunk_size = 200000  # Can tune this later for accuracy.

while start < len(books_full_unmatched_uncleaned):
    while True:
        end = min(start + chunk_size, len(books_full_unmatched_uncleaned))
        chunk = books_full_unmatched_uncleaned.iloc[start:end]
        temp_file = f'{base_filename}_{file_index}.csv'
        
        chunk.to_csv(temp_file, index=False)

        # Check file size
        if os.path.getsize(temp_file) <= max_size_bytes or chunk_size == 1:
            break
        else:
            # Reduce size and try again
            print("reduce size and try again")
            print("old size = ", chunk_size, " new size = ", chunk_size * 0.7)
            chunk_size = int(chunk_size * 0.7)

    print(f'saved {temp_file} with {len(chunk)} rows')
    start += len(chunk)
    file_index += 1