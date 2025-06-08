import pandas as pd
import os
# --- Configuration ---

BASE_FILENAME = "books_matched_no_dupe_title_author_pairs"

# --- Combine ---

def combine_csv_files(base_filename: str):
    """
    Combines multiple CSV files with a common base filename into a single DataFrame.

    Args:
        base_filename (str): The base name of the CSV files (e.g., 'books_matched').
                             The function expects files like 'books_matched_1.csv',
                             'books_matched_2.csv', etc.

    Returns:
        pd.DataFrame: A DataFrame containing the combined data from all found CSVs.
                      Returns an empty DataFrame if no files are found.
    """
    combined_df = pd.DataFrame()
    file_index = 1

    print(f"Attempting to combine files starting with '{base_filename}_*.csv'...")

    while True:
        filename = f'{base_filename}_{file_index}.csv'
        if not os.path.exists(filename):
            print(f"No more files found. Stopped at '{filename}'.")
            break

        print(f'Loading {filename}...')
        try:
            temp_df = pd.read_csv(filename)
            combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            pass

        file_index += 1

    if not combined_df.empty:
        print(f'Successfully combined {file_index - 1} files.')
        print(f'Combined DataFrame shape: {combined_df.shape}')
        output_filename = f"{base_filename}.csv"
        combined_df.to_csv(output_filename, index=False) # index=False prevents writing DataFrame index
        print(f'Combined data saved to {output_filename}')
    else:
        print("No CSV files were found or combined.")

    return combined_df

if __name__ == "__main__":
    combine_csv_files(BASE_FILENAME)