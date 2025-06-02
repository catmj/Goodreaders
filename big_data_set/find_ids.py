import ijson
import csv
import sys
import json

def extract_goodreads_data_to_csv_large_file(json_file_path, csv_file_path):
    """
    Extracts Goodreads URLs (cleaned), Book IDs, Titles (shortened by removing text after first colon),
    and Author IDs from a large JSON file and writes them to a CSV file.

    It first attempts to process the file as JSON Lines (NDJSON).
    If that fails, it falls back to using ijson for single large JSON arrays or objects.

    Args:
        json_file_path (str): The path to your input JSON file.
        csv_file_path (str): The path where you want to save the output CSV file.
    """
    goodreads_entries_found = 0
    url_prefix_to_remove = "https://www.goodreads.com/book/show/"

    try:
        # --- FIRST ATTEMPT: Assume JSON Lines (NDJSON) ---
        print(f"Attempting to process '{json_file_path}' as JSON Lines...", file=sys.stderr)
        with open(json_file_path, 'r', encoding='utf-8') as f_json_text:
            with open(csv_file_path, 'w', newline='', encoding='utf-8') as f_csv:
                csv_writer = csv.writer(f_csv)
                # Updated CSV Header
                csv_writer.writerow(['Cleaned Goodreads URL', 'Book ID', 'Shortened Title', 'Author ID'])

                processed_lines_count = 0
                for line in f_json_text:
                    processed_lines_count += 1
                    try:
                        item = json.loads(line)
                        original_url = item.get("url", "")
                        book_id = item.get("book_id", "")
                        original_title_without_series = item.get("title_without_series", "")
                        author_id = ""

                        if "authors" in item and item["authors"] and isinstance(item["authors"], list):
                            author_id = item["authors"][0].get("author_id", "")

                        # Clean the URL
                        cleaned_url = ""
                        if original_url.startswith(url_prefix_to_remove):
                            cleaned_url = original_url.replace(url_prefix_to_remove, "", 1)

                        # *** Shorten the title ***
                        shortened_title = original_title_without_series.split(':', 1)[0].strip()

                        if cleaned_url and book_id and shortened_title:
                            csv_writer.writerow([cleaned_url, book_id, shortened_title, author_id])
                            goodreads_entries_found += 1
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed JSON line {processed_lines_count}: {line.strip()} - Error: {e}", file=sys.stderr)
                    except Exception as e:
                        print(f"Warning: An unexpected error occurred processing line {processed_lines_count}: {line.strip()} - Error: {e}", file=sys.stderr)
                print(f"Finished JSON Lines attempt. Extracted {goodreads_entries_found} entries.", file=sys.stderr)

                if goodreads_entries_found > 0 or processed_lines_count > 0:
                    print(f"Successfully processed {processed_lines_count} lines (JSON Lines format).", file=sys.stderr)
                    return

    except FileNotFoundError:
        print(f"Error: JSON file not found at '{json_file_path}'", file=sys.stderr)
        return
    except UnicodeDecodeError as e:
        print(f"Error: Encoding issue while reading '{json_file_path}'. Try checking file encoding. Error: {e}", file=sys.stderr)
        return
    except Exception as e:
        print(f"JSON Lines initial attempt failed for an unexpected reason: {e}. Falling back to ijson...", file=sys.stderr)

    # --- FALLBACK: Use ijson for single large JSON arrays or objects ---
    goodreads_entries_found = 0

    try:
        print(f"Attempting to process '{json_file_path}' with ijson (assuming array or single object structure)...", file=sys.stderr)
        with open(json_file_path, 'rb') as f_json:
            with open(csv_file_path, 'w', newline='', encoding='utf-8') as f_csv:
                csv_writer = csv.writer(f_csv)
                # Updated CSV Header for ijson path
                csv_writer.writerow(['Cleaned Goodreads URL', 'Book ID', 'Shortened Title', 'Author ID'])

                item_prefix = None
                f_json.seek(0)

                try:
                    objects_peek = ijson.items(f_json, 'item')
                    next(objects_peek)
                    item_prefix = 'item'
                except StopIteration:
                    f_json.seek(0)
                    objects_peek = ijson.items(f_json, '')
                    try:
                        first_item_peek = next(objects_peek)
                        if isinstance(first_item_peek, dict) and "isbn" in first_item_peek and "url" in first_item_peek:
                            item_prefix = ''
                    except StopIteration:
                        pass

                f_json.seek(0)

                if item_prefix is None:
                    print(f"Could not determine ijson parsing path for '{json_file_path}'. File might be empty or malformed differently.", file=sys.stderr)
                    return

                parser = ijson.items(f_json, item_prefix)
                for item_data in parser:
                    if isinstance(item_data, dict):
                        original_url = item_data.get("url", "")
                        book_id = item_data.get("book_id", "")
                        original_title_without_series = item_data.get("title_without_series", "")
                        author_id = ""

                        if "authors" in item_data and item_data["authors"] and isinstance(item_data["authors"], list):
                            author_id = item_data["authors"][0].get("author_id", "")

                        # Clean the URL
                        cleaned_url = ""
                        if original_url.startswith(url_prefix_to_remove):
                            cleaned_url = original_url.replace(url_prefix_to_remove, "", 1)

                        # *** Shorten the title ***
                        shortened_title = original_title_without_series.split(':', 1)[0].strip()

                        if cleaned_url and book_id and shortened_title:
                            csv_writer.writerow([cleaned_url, book_id, shortened_title, author_id])
                            goodreads_entries_found += 1

    except FileNotFoundError:
        print(f"Error: JSON file not found at '{json_file_path}'", file=sys.stderr)
        return
    except ijson.common.JSONError as e:
        print(f"Error parsing JSON with ijson from '{json_file_path}'. Check file integrity. Error: {e}", file=sys.stderr)
        return
    except Exception as e:
        print(f"An unexpected error occurred during ijson processing: {e}", file=sys.stderr)
        return

    print(f"Successfully extracted {goodreads_entries_found} Goodreads entries to '{csv_file_path}' (ijson method).", file=sys.stderr)

# --- How to use the function ---
json_input_file = 'goodreads_books.json' # Your actual file name
csv_output_file = 'goodreads_book_data_v4.csv'

extract_goodreads_data_to_csv_large_file(json_input_file, csv_output_file)