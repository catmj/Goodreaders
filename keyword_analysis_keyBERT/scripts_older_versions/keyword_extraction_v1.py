# using https://github.com/MaartenGr/KeyBERT
# to run first do: pip install -r requirements.txt


# PACKAGES
import pandas as pd # type: ignore
from keybert import KeyBERT # type: ignore


# IMPORTING COPIES OF DATA
df1 = pd.read_csv('books_matched_copy.csv')
df2 = pd.read_csv('reviews_matched_copy.csv')
df3 = pd.read_csv('preprocessing_books.csv')

kw_model = KeyBERT()


def extract(text,n=10):
    
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english',
                              use_mmr=True, diversity=0.2, nr_candidates=20, top_n=n)

    keys = []

    for key in keywords:
        keys.append(key[0])

    key_string = ", ".join(keys) 
       
    #print(key_string)
    return key_string




# KEYWORDS FOR BOOKS.CSV
keyword_lists_df1 = df1['description'].astype(str).apply(lambda x:extract(x)) # Create keyword lists.
#print(keyword_lists_df1)
keyword_lists_df1.to_csv('output_file_books.csv', index=True) # Convert keyword lists to .csv file.

# KEYWORDS FOR BOOKS.CSV
keyword_lists_df3 = df3['description'].astype(str).apply(lambda x:extract(x)) # Create keyword lists.
#print(keyword_lists_df1)
keyword_lists_df3.to_csv('output_file_preprocessed_books.csv', index=True) # Convert keyword lists to .csv file.

# KEYWORDS FOR REVIEWS.CSV
keyword_lists_df2 = df2['review'].astype(str).apply(lambda x:extract(x)) # Create keyword lists.
# print(keyword_lists_df2)
keyword_lists_df2.to_csv('output_file_reviews.csv', index=True) # Convert keyword lists to .csv file.

