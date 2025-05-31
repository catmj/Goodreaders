import pandas as pd # type: ignore
from sentence_transformers import SentenceTransformer, util # type: ignore
import time


model = SentenceTransformer("all-MiniLM-L6-v2")

key_frame = pd.read_csv('output_file_books_15.csv')

key_sentences = key_frame.description.tolist()

keys = set()

for key_sentence in key_sentences:
  if type(key_sentence) == str:
    keys = keys.union(set(key_sentence.split(", ")))

keys = list(keys)
print("Encode the keywords.")
key_embeddings = model.encode(keys, batch_size=64, show_progress_bar=True, convert_to_tensor=True)


print("Start clustering")
start_time = time.time()

# Two parameters to tune:
# min_cluster_size: Only consider cluster that have at least 10 elements
# threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar

clusters = util.community_detection(key_embeddings, min_community_size=10, threshold=0.6)

print(f"Clustering done after {time.time() - start_time:.2f} sec")

total_clustered = 0

keys_by_cluster = []

# Print for all clusters the top 3 and bottom 3 elements
for i, cluster in enumerate(clusters):
    print(f"\nCluster {i + 1}, #{len(cluster)} Elements ")
    cluster_sentence = []
    for key_id in cluster:
        print("\t", keys[key_id])
        total_clustered +=1
        cluster_sentence.append(keys[key_id])
    keys_by_cluster.append(", ".join(cluster_sentence))

print(total_clustered)
print(len(keys))

df = pd.DataFrame({"key_clusters":keys_by_cluster})
df.to_csv('keys_by_cluster.csv')
