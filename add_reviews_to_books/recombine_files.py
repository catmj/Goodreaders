import pandas as pd
import os

# Base filename pattern
base_filename = './books_full_unmatched_uncleaned'
combined_df = pd.DataFrame()

file_index = 1

# Loop until no more files are found
while True:
    filename = f'{base_filename}_{file_index}.csv'
    if not os.path.exists(filename):
        break  # Exit loop if file doesn't exist

    print(f'Loading {filename}...')
    temp_df = pd.read_csv(filename)
    combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

    file_index += 1

print(f'Combined DataFrame shape: {combined_df.shape}')
combined_df.to_csv('books_full_unmatched_uncleaned.csv')