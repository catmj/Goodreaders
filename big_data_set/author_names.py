

import ijson
import csv
import json
import sys
import os

def extract_author_data_to_csv(author_json_path, output_csv_path):
    """
    Extracts author IDs and names from a large JSON file and writes them to a CSV file.
    It attempts to handle JSON Lines, or a single JSON object/array of author entries.

    Args:
        author_json_path (str): The path to your input JSON file containing author data.
        output_csv_path (str): The path where you want to save the output CSV file.
    """
    authors_extracted = 0

    if not os.path.exists(author_json_path):
        print(f"Error: Author JSON file not found at '{author_json_path}'", file=sys.stderr)
        return

    try:
        # --- FIRST ATTEMPT: Assume JSON Lines (NDJSON) ---
        print(f"Attempting to process '{author_json_path}' as JSON Lines...", file=sys.stderr)
        with open(author_json_path, 'r', encoding='utf-8') as f_json_text:
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as f_csv:
                csv_writer = csv.writer(f_csv)
                csv_writer.writerow(['Author ID', 'Author Name']) # CSV Header

                processed_lines_count = 0
                for line in f_json_text:
                    processed_lines_count += 1
                    try:
                        item = json.loads(line) # Parse each line as a separate JSON object
                        author_id = item.get("author_id")
                        author_name = item.get("name")
                        
                        if author_id and author_name:
                            csv_writer.writerow([str(author_id), author_name])
                            authors_extracted += 1
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed JSON line {processed_lines_count} in author file. Error: {e}", file=sys.stderr)
                    except Exception as e:
                        print(f"Warning: An unexpected error occurred processing author line {processed_lines_count}: {line.strip()} - Error: {e}", file=sys.stderr)
                
                print(f"Finished JSON Lines attempt for author data. Extracted {authors_extracted} entries.", file=sys.stderr)

                if authors_extracted > 0 or processed_lines_count > 0:
                    print(f"Successfully processed {processed_lines_count} lines (JSON Lines format) for author data.", file=sys.stderr)
                    return # Exit the function, as we're done

    except UnicodeDecodeError as e:
        print(f"Error: Encoding issue while reading author JSON '{author_json_path}'. Error: {e}", file=sys.stderr)
        # Fall through to ijson attempt if text decoding fails
    except Exception as e:
        print(f"JSON Lines attempt for author data failed: {e}. Falling back to ijson...", file=sys.stderr)

    # --- FALLBACK: Use ijson for single large JSON arrays or objects ---
    authors_extracted = 0 # Reset counter for ijson attempt

    try:
        print(f"Attempting to process '{author_json_path}' with ijson...", file=sys.stderr)
        with open(author_json_path, 'rb') as f_json: # Open in binary mode for ijson
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as f_csv:
                csv_writer = csv.writer(f_csv)
                csv_writer.writerow(['Author ID', 'Author Name']) # CSV Header for ijson path

                item_prefix = None
                f_json.seek(0) # Ensure at beginning for peek

                try:
                    # Try as an array of author objects: [{"author_id": "...", "name": "..."}, ...]
                    objects_peek = ijson.items(f_json, 'item')
                    next(objects_peek) # Try to get the first item
                    item_prefix = 'item'
                except StopIteration:
                    f_json.seek(0)
                    try:
                        # Try as a dictionary where values are author objects: {"author_id_1": {...}, "author_id_2": {...}}
                        objects_peek = ijson.items(f_json, '')
                        first_item_peek = next(objects_peek)
                        if isinstance(first_item_peek, dict) and "author_id" in first_item_peek and "name" in first_item_peek:
                            item_prefix = ''
                        else:
                            # If the first item is not directly an author object,
                            # try looking under a 'root' key if it exists and contains author objects.
                            f_json.seek(0)
                            objects_peek = ijson.items(f_json, 'root') # For {"root": {"author_id": ..., "name": ...}}
                            first_item_peek = next(objects_peek)
                            if isinstance(first_item_peek, dict) and "author_id" in first_item_peek and "name" in first_item_peek:
                                item_prefix = 'root'
                    except StopIteration:
                        pass # Still not found

                f_json.seek(0) # Reset file pointer for actual parsing

                if item_prefix is None:
                    print(f"Could not determine ijson parsing path for author JSON '{author_json_path}'. File might be empty or malformed differently.", file=sys.stderr)
                    return

                parser = ijson.items(f_json, item_prefix)
                for item_data in parser:
                    if isinstance(item_data, dict):
                        author_id = item_data.get("author_id")
                        author_name = item_data.get("name")
                        if author_id and author_name:
                            csv_writer.writerow([str(author_id), author_name])
                            authors_extracted += 1
        
        print(f"Successfully loaded {authors_extracted} author entries (ijson method).", file=sys.stderr)

    except ijson.common.JSONError as e:
        print(f"Error parsing author JSON with ijson from '{author_json_path}'. Check file integrity. Error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during ijson processing of author data: {e}", file=sys.stderr)

# --- How to use the function ---
author_json_file = 'goodreads_book_authors.json' # <<< IMPORTANT: Replace with your actual author JSON file name
output_author_csv = 'goodreads_author_ids_and_names.csv'

extract_author_data_to_csv(author_json_file, output_author_csv)