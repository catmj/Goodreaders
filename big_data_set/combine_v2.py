import pandas as pd
import sys
import os
import re

def clean_title(title):
    """
    Cleans and standardizes a book title for matching purposes,
    then truncates to the first 61 characters.
    - Converts to string, handles None/NaN.
    - Removes common subtitle separators (colon, parentheses) and text after them.
    - Converts to lowercase.
    - Removes non-alphanumeric characters (keeping spaces).
    - Replaces multiple spaces with a single space and strips extra whitespace.
    - Truncates to the first 61 characters.
    """
    if pd.isna(title):
        return ""
    title = str(title)
    
    # Remove text after first colon or opening parenthesis (for subtitles)
    title = re.split(r'[:(]', title, 1)[0]
    
    # Convert to lowercase
    title = title.lower()
    
    # Remove non-alphanumeric characters (keep spaces)
    title = re.sub(r'[^a-z0-9\s]', '', title)
    
    # Replace multiple spaces with a single space and strip leading/trailing spaces
    title = re.sub(r'\s+', ' ', title).strip()
    
    # Truncate to the first 61 characters for matching
    title = title[:61]
    
    return title

def clean_author_name(author_name):
    """
    Cleans and standardizes an author name for matching purposes (full match).
    - Converts to string, handles None/NaN.
    - Converts to lowercase.
    - Removes non-alphanumeric characters (keeping spaces).
    - Strips extra whitespace.
    """
    if pd.isna(author_name):
        return ""
    author_name = str(author_name)
    
    # Convert to lowercase
    author_name = author_name.lower()
    
    # Remove non-alphanumeric characters (keep spaces)
    author_name = re.sub(r'[^a-z0-9\s]', '', author_name)
    
    # Replace multiple spaces with a single space and strip leading/trailing spaces
    author_name = re.sub(r'\s+', ' ', author_name).strip()
    
    return author_name

def add_goodreads_url_by_title_author(target_csv_path, goodreads_source_csv_path, output_csv_path):
    """
    Adds Goodreads URLs to a CSV file by matching on cleaned Title (first 10 chars)
    and full cleaned Author Name. It then removes entries for which a Goodreads URL
    could not be found.

    Args:
        target_csv_path (str): Path to your CSV file needing URLs (e.g., 'reviews_unmatched.csv').
        goodreads_source_csv_path (str): Path to a CSV with Goodreads URLs, Titles, and Authors
                                          (e.g., generated from Goodreads JSON data).
        output_csv_path (str): Path where the new CSV with Goodreads URLs will be saved.
    """
    if not os.path.exists(target_csv_path):
        print(f"Error: Target CSV file not found at '{target_csv_path}'. Please ensure it exists.", file=sys.stderr)
        return
    if not os.path.exists(goodreads_source_csv_path):
        print(f"Error: Goodreads Source CSV file not found at '{goodreads_source_csv_path}'. Please ensure it exists.", file=sys.stderr)
        return

    try:
        # Load the CSV file that needs Goodreads URLs
        df_target = pd.read_csv(target_csv_path, encoding='utf-8')
        print(f"Loaded target CSV: {target_csv_path}. Rows: {len(df_target)}", file=sys.stderr)
        print(f"Target CSV columns: {df_target.columns.tolist()}", file=sys.stderr)

        # Load the Goodreads source data
        df_source = pd.read_csv(goodreads_source_csv_path, encoding='utf-8')
        print(f"Loaded Goodreads source CSV: {goodreads_source_csv_path}. Rows: {len(df_source)}", file=sys.stderr)
        print(f"Source CSV columns: {df_source.columns.tolist()}", file=sys.stderr)
        
        # --- Apply Cleaning to Matching Columns ---
        # Clean 'Title' and 'Author' columns in df_target
        if 'Title' in df_target.columns:
            df_target['Cleaned_Title_For_Merge'] = df_target['Title'].apply(clean_title)
        else:
            print(f"Warning: 'Title' column not found in {target_csv_path}. Matching might be affected.", file=sys.stderr)
            df_target['Cleaned_Title_For_Merge'] = "" 
        
        if 'Author' in df_target.columns:
            df_target['Cleaned_Author_For_Merge'] = df_target['Author'].apply(clean_author_name)
        else:
            print(f"Warning: 'Author' column not found in {target_csv_path}. Matching might be affected.", file=sys.stderr)
            df_target['Cleaned_Author_For_Merge'] = "" 

        # Clean 'Shortened Title' and 'Author Name' columns in df_source
        if 'Shortened Title' in df_source.columns:
            df_source['Cleaned_Title_For_Merge'] = df_source['Shortened Title'].apply(clean_title)
        elif 'Title' in df_source.columns: # Fallback if source uses 'Title' instead of 'Shortened Title'
            df_source['Cleaned_Title_For_Merge'] = df_source['Title'].apply(clean_title)
        else:
            print(f"Error: Neither 'Shortened Title' nor 'Title' found in {goodreads_source_csv_path}. Cannot proceed with title matching.", file=sys.stderr)
            return
            
        if 'Author Name' in df_source.columns:
            df_source['Cleaned_Author_For_Merge'] = df_source['Author Name'].apply(clean_author_name)
        elif 'Author' in df_source.columns: # Fallback if source uses 'Author' instead of 'Author Name'
            df_source['Cleaned_Author_For_Merge'] = df_source['Author'].apply(clean_author_name)
        else:
            print(f"Error: Neither 'Author Name' nor 'Author' found in {goodreads_source_csv_path}. Cannot proceed with author matching.", file=sys.stderr)
            return

        # Select only the necessary columns from the source DataFrame for merging
        df_source_for_merge = df_source[['Cleaned Goodreads URL', 'Cleaned_Title_For_Merge', 'Cleaned_Author_For_Merge']].drop_duplicates(
            subset=['Cleaned_Title_For_Merge', 'Cleaned_Author_For_Merge']
        )
        print(f"Goodreads source data prepared for merge (unique cleaned title/author pairs): {len(df_source_for_merge)}", file=sys.stderr)

        # Perform the left merge
        merged_df = pd.merge(
            df_target,
            df_source_for_merge,
            left_on=['Cleaned_Title_For_Merge', 'Cleaned_Author_For_Merge'],
            right_on=['Cleaned_Title_For_Merge', 'Cleaned_Author_For_Merge'],
            how='left'
        )

        # Drop the temporary cleaned columns
        merged_df = merged_df.drop(columns=['Cleaned_Title_For_Merge', 'Cleaned_Author_For_Merge'])

        # --- NEW: Filter out rows where 'Cleaned Goodreads URL' is missing (NaN) ---
        initial_rows = len(merged_df)
        merged_df = merged_df.dropna(subset=['Cleaned Goodreads URL'])
        rows_after_filter = len(merged_df)
        print(f"Removed {initial_rows - rows_after_filter} rows with missing 'Cleaned Goodreads URL'.", file=sys.stderr)


        # Reorder columns to place Cleaned Goodreads URL near relevant info (e.g., after Author)
        cols = merged_df.columns.tolist()
        if 'Author' in cols and 'Cleaned Goodreads URL' in cols:
            author_idx = cols.index('Author')
            if 'Cleaned Goodreads URL' in cols: 
                cols.remove('Cleaned Goodreads URL')
            cols.insert(author_idx + 1, 'Cleaned Goodreads URL')
        
        merged_df = merged_df[cols]

        # Save the result to a new CSV
        merged_df.to_csv(output_csv_path, index=False, encoding='utf-8')

        print(f"\nSuccessfully added Goodreads URLs to '{output_csv_path}'.", file=sys.stderr)
        print(f"Total entries in output file: {len(merged_df)}", file=sys.stderr)
        print(f"Number of matches found (rows with Cleaned Goodreads URL): {merged_df['Cleaned Goodreads URL'].count()}", file=sys.stderr)
        print("Final columns in the new CSV:", merged_df.columns.tolist(), file=sys.stderr)

    except FileNotFoundError as e:
        print(f"Error: A file was not found. {e}", file=sys.stderr)
    except pd.errors.EmptyDataError:
        print(f"Error: One of the input CSV files is empty or has no columns.", file=sys.stderr)
    except KeyError as e:
        print(f"Error: Missing expected column(s). Please check column names in your CSVs. Error: {e}", file=sys.stderr)
        if 'df_target' in locals(): print(f"Target CSV columns: {df_target.columns.tolist()}", file=sys.stderr)
        if 'df_source' in locals(): print(f"Source CSV columns: {df_source.columns.tolist()}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)

# --- How to use the function ---
# IMPORTANT: Replace these with your actual file names/paths

# This is your CSV file with Title and Author, where you want to add Goodreads URLs.
your_input_csv = 'reviews_unmatched.csv' 

# This is the CSV file that contains the Goodreads URLs, Titles, and Authors for matching.
# For example, it might be named 'goodreads_book_data_shortened_titles.csv'
# from our previous steps.
goodreads_urls_source_csv = 'final_combined_book_data.csv' 

# The name for the new CSV file with Goodreads URLs added and unmatched rows removed
output_csv_with_urls = 'reviews_with_goodreads_urls_filtered.csv'

# Call the function to add Goodreads URLs and filter the results
add_goodreads_url_by_title_author(your_input_csv, goodreads_urls_source_csv, output_csv_with_urls)