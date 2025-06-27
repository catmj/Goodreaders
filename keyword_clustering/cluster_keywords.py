import pandas as pd # type: ignore
from sentence_transformers import SentenceTransformer, util # type: ignore
import time
import nltk
from nltk.corpus import words # Import 'words' corpus
import os

# --- Configuration Variables ---
INPUT_FILE_PATH = "../keyword_analysis_keyBERT/output_file_books_train_preprocessed_n15_div0.2.csv"
OUTPUT_FILE_PATH = "keys_by_cluster.csv"
MIN_COMMUNITY_SIZE = 4
THRESHOLD = 0.65
# --- End Configuration Variables ---

nltk.download('words')


model = SentenceTransformer("all-MiniLM-L6-v2")

key_frame = pd.read_csv(INPUT_FILE_PATH)

key_sentences = key_frame.keywords.tolist()

keys = set()

for key_sentence in key_sentences:
  if isinstance(key_sentence, str): # Use isinstance for type checking
    keys = keys.union(set(key_sentence.split(", ")))

# --- Filter out non-English words ---
print("Filtering non-English words...")
# Ensure english_words is loaded *after* confirming the download
english_words = set(words.words())
filtered_keys = []
for key in keys:
    # Convert to lowercase for consistent checking against the English dictionary
    processed_key = key.lower().strip() 
    
    # Check if all words in the keyphrase are English
    # This handles multi-word keyphrases (e.g., "science fiction")
    # Also, ensure non-empty strings are processed
    if processed_key: # Check if the processed key is not empty
        is_english_keyphrase = all(word in english_words for word in processed_key.split())
    else:
        is_english_keyphrase = False # Treat empty strings as non-English
    
    # Keep words longer than 2 characters and that are identified as English
    if is_english_keyphrase and len(processed_key) > 2:
        filtered_keys.append(key) # Append the original key

keys = list(filtered_keys) # Update the 'keys' list with only English words
# --- END Filtering ---

print(f"Number of keywords after filtering: {len(keys)}")
print(keys[:10])

print("Encode the keywords.")
key_embeddings = model.encode(keys, batch_size=64, show_progress_bar=True, convert_to_tensor=True)


print("Start clustering")
start_time = time.time()

# Two parameters to tune:
# min_cluster_size: Only consider cluster that have at least 10 elements
# threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar

clusters = util.community_detection(key_embeddings, min_community_size=MIN_COMMUNITY_SIZE, threshold=THRESHOLD)

print(f"Clustering done after {time.time() - start_time:.2f} sec")

total_clustered = 0

keys_by_cluster = []


for i, cluster in enumerate(clusters):
    print(f"\nCluster {i + 1}, #{len(cluster)} Elements ")
    cluster_sentence = []
    for key_id in cluster:
        print("\t", keys[key_id])
        total_clustered +=1
        cluster_sentence.append(keys[key_id])
    keys_by_cluster.append(", ".join(cluster_sentence))

print(f"Total clustered keywords: {total_clustered}")
print(f"Total unique keywords processed: {len(keys)}")

df = pd.DataFrame({"key_clusters":keys_by_cluster})

df.to_csv(OUTPUT_FILE_PATH, index=False)
print(f"Keyword clusters saved to '{OUTPUT_FILE_PATH}'")