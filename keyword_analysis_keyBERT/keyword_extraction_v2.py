# using https://github.com/MaartenGr/KeyBERT
# to run first do: pip install -r requirements.txt


# PACKAGES
import pandas as pd # type: ignore
from keybert import KeyBERT # type: ignore
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# IMPORTING COPIES OF DATA
df = pd.read_csv('../cleaned_data_2/books_matched.csv')

# PRE-PROCESSING
stop_words = set(stopwords.words('english'))
# Choose stopwords to exclude:
new_stop_words = ["book", "books", "novel", "novels", "writer", "writers", "write", "writes", "writing", "written", "author", "authors", "edition", "editions",
                  "bestseller", "bestsellers", "bestselling", "reader", "readers", "reading", "read", "reads", "isbn", "title", "titles", "titled", "titular",   
                  "genre", "genres", "review", "reviews", "reviewed", "reviewer", "reviewers", "critic", "critics", "series", "newest", "latest", "literary", 
                  "hardcover", "paperback", "publisher", "publishers", "published", "yet"]
# Add the above words to the list of default stopwords:
stop_words = list(stop_words.union(new_stop_words))
def pre_process(text):
    # Make text lowercase.
    text = text.lower()
    # Remove special characters and digits.
    text = re.sub("(\\d|\\W)+"," ",text)
    # May also need to remove tags?
    text = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    # Convert text from string to list.
    text = text.split()
    # Remove stopwords from text.
    text = [word for word in text if word not in stop_words]
    # Remove short words (less than 3 letters).
    text = [word for word in text if len(word) >= 3]
    # Lemmatize.
    lmtzr = WordNetLemmatizer()
    text = [lmtzr.lemmatize(word) for word in text]
    return ' '.join(text)

# PREPROCESSING FOR BOOKS.CSV
df_preprocessed = df['description'].astype(str).apply(lambda x:pre_process(x)) # Preprocess


####Use KeyBERT

kw_model = KeyBERT()


def extract(text,n=10, diversity = 0.2):
    
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english',
                              use_mmr=True, diversity=diversity, nr_candidates=20, top_n=n)

    keys = []

    for key in keywords:
        keys.append(key[0])

    key_string = ", ".join(keys) 
       
    #print(key_string)
    return key_string




# KEYWORDS FOR BOOKS.CSV
keyword_lists_df = df['description'].astype(str).apply(lambda x:extract(x)) # Create keyword lists.
#print(keyword_lists_df1)
keyword_lists_df.to_csv('output_file_books.csv', index=True) # Convert keyword lists to .csv file.

# KEYWORDS FOR BOOKS.CSV
keyword_lists_df_preprocessed = df_preprocessed['description'].astype(str).apply(lambda x:extract(x)) # Create keyword lists.
#print(keyword_lists_df1)
keyword_lists_df_preprocessed.to_csv('output_file_preprocessed_books.csv', index=True) # Convert keyword lists to .csv file.
