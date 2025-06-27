import pandas as pd
from keybert import KeyBERT
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os


nltk.download('stopwords')
nltk.download('wordnet')

class KeywordExtractor:
    """
    A class to encapsulate the text preprocessing and keyword extraction logic
    for multiple CSV files with varying KeyBERT parameters, retaining key metadata.
    """
    def __init__(self, file_paths):
        if not isinstance(file_paths, list) or not file_paths:
            raise ValueError("file_paths must be a non-empty list of file paths.")
        
        self.file_paths = file_paths
        self.stop_words = self._load_and_extend_stopwords()
        self.lemmatizer = WordNetLemmatizer()
        self.kw_model = KeyBERT()

    def _load_and_extend_stopwords(self):
        default_stopwords = set(stopwords.words('english'))
        custom_stopwords = [
            "book", "books", "novel", "novels", "writer", "writers", "write", "writes",
            "writing", "written", "author", "authors", "edition", "editions",
            "bestseller", "bestsellers", "bestselling", "reader", "readers",
            "reading", "read", "reads", "isbn", "title", "titles", "titled",
            "titular", "genre", "genres", "review", "reviews", "reviewed",
            "reviewer", "reviewers", "critic", "critics", "series", "newest",
            "latest", "literary", "hardcover", "paperback", "publisher",
            "publishers", "published", "yet"
        ]
        return list(default_stopwords.union(custom_stopwords))

    def preprocess_text(self, text):
        text = str(text).lower()
        text = re.sub(r"(\d|\W)+", " ", text)
        text = re.sub(r"&lt;/?.*?&gt;", " ", text)

        words = text.split()
        words = [word for word in words if word not in self.stop_words and len(word) >= 3]
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)

    def extract_keywords(self, text, n=10, diversity=0.2):
        if not text.strip():
            return ""

        keywords = self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 1),
            stop_words='english',
            use_mmr=True,
            diversity=diversity,
            nr_candidates=20,
            top_n=n
        )
        return ", ".join([key[0] for key in keywords])

    def process_file(self, file_path, param_combinations):
        """
        Processes a single CSV file: loads, preprocesses, extracts keywords,
        and saves results including specified metadata columns.
        """
        print(f"\nProcessing file: {file_path}")
        try:
            df = pd.read_csv(file_path)
            
            # Define required columns for processing and metadata
            #required_cols = ['description']
            metadata_cols = ['genres', 'title', 'author_name', 'rating', 'num_ratings']
            
            # Check for required 'description' column
            if 'description' not in df.columns:
                print(f"Skipping {file_path}: 'description' column not found.")
                return

            # Check for optional metadata columns and warn if missing
            available_metadata_cols = [col for col in metadata_cols if col in df.columns]
            missing_metadata_cols = [col for col in metadata_cols if col not in df.columns]
            if missing_metadata_cols:
                print(f"Warning: Following metadata columns missing in {file_path}: {', '.join(missing_metadata_cols)}. They will be omitted from output.")

            # Get base name for output files
            base_name = os.path.basename(file_path).replace('.csv', '')

            print("Starting text preprocessing...")
            df['description_preprocessed'] = df['description'].astype(str).apply(self.preprocess_text)
            print("Preprocessing complete.")

            # Loop through each combination of n and diversity
            for n_val, diversity_val in param_combinations:
                print(f"\n--- Extracting keywords for n={n_val}, diversity={diversity_val} ---")
                
                # Prepare base columns for output (metadata + description)
                output_base_df = df[available_metadata_cols + ['description']].copy()

                # Extract from preprocessed descriptions
                col_name_preprocessed_keywords = 'keywords'
                output_base_df[col_name_preprocessed_keywords] = df['description_preprocessed'].astype(str).apply(
                    lambda x: self.extract_keywords(x, n=n_val, diversity=diversity_val)
                )
                output_preprocessed_path = f'output_file_{base_name}_preprocessed_n{n_val}_div{diversity_val}.csv'
                
                # Select columns for output: original metadata + original description + new keyword column
                cols_to_save_preprocessed = available_metadata_cols + [col_name_preprocessed_keywords]
                output_base_df[cols_to_save_preprocessed].to_csv(output_preprocessed_path, index=False)
                print(f"Keywords from preprocessed descriptions saved to '{output_preprocessed_path}'")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    def run_keyword_extraction_for_all_files(self, param_combinations):
        """
        Iterates through all provided file paths and processes each one
        with the given parameter combinations.
        """
        for file_path in self.file_paths:
            self.process_file(file_path, param_combinations)

if __name__ == "__main__":
    data_folder_path = 'data_folder' 
    
    # Define ranges for n and diversity
    n_values = [15]
    diversity_values = [0.2]

    # Generate all combinations (grid search)
    keyword_params_grid = [(n, div) for n in n_values for div in diversity_values]

    if os.path.exists(data_folder_path):
        all_csv_files_in_folder = [os.path.join(data_folder_path, f) for f in os.listdir(data_folder_path) if f.endswith('.csv')]
        if not all_csv_files_in_folder:
            print(f"No CSV files found in {data_folder_path}. Please check the path and content.")
            exit()
        data_files = all_csv_files_in_folder
    else:
        print(f"Error: Data folder '{data_folder_path}' not found. Please create it and place your CSVs, or adjust the path.")
        exit()

    extractor = KeywordExtractor(data_files)
    extractor.run_keyword_extraction_for_all_files(keyword_params_grid)