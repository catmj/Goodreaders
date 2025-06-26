import streamlit as st
import pandas as pd
import os
import sys # Import sys module for path manipulation

# --- Import streamlit-searchbox (assuming it's in requirements.txt or pre-installed) ---
from streamlit_searchbox import st_searchbox


# --- Adjust Python Path for Module Discovery ---
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
# Add the project root to sys.path so Python can find 'combined_model' as a top-level package
sys.path.insert(0, project_root)

# --- IMPORT RECOMMENDATION FUNCTIONS FROM YOUR LOCAL FILE ---
try:
    from combined_model.combined_get_recs_final import (
        get_combined_recommendations,
        recommend_by_multiplying_scores, # Only importing this specific recommendation strategy
    )
except ImportError as e:
    # If the import fails, display an message and stop the Streamlit application.
    st.error(f"Error loading recommendation functions: {e}. Please ensure 'combined_model' directory is at the project root level, contains 'combined.py', and potentially has an '__init__.py' file. Also, verify internal imports within 'combined.py' for modules like 'cb_get_recs' and 'cf_get_recs' are relative (e.g., 'from .cb_get_recs import ...'). Error: {e}")
    st.stop() # Stop the app if functions cannot be loaded


# --- Predetermined List of Books (Original Lowercase Format for Backend) ---
# Define the absolute path to the book list file by joining it with the project_root.
book_list_file_path_abs = os.path.join(project_root, "collab_filtering/trained_data/book_list_features_600_lambda_5.txt")
predetermined_book_list_backend_format = []

try:
    # Open and read the book list file, adding each line (stripped of whitespace) to the list.
    with open(book_list_file_path_abs, 'r', encoding='utf-8') as f:
        for line in f:
            predetermined_book_list_backend_format.append(line.strip()) # Read each line and strip whitespace
except FileNotFoundError:
    # Handle the case where the book list file does not exist.
    st.error(f"Error: The book list file was not found at `{book_list_file_path_abs}`. Please check the path and ensure the file exists at the project root level within the `collab_filtering/trained_data/` directory.")
    st.stop() # Stop the app if the book list cannot be loaded
except Exception as e:
    # Handle any other exceptions that might occur during file loading.
    st.error(f"An error occurred while loading the book list from `{book_list_file_path_abs}`: {e}")
    st.stop()


# Function to format book titles for display in the Streamlit UI.
# This function handles capitalization, quotes, and author/title separation.
def format_book_title_for_display(title_str: str) -> str:
    # First, handle outer quotes if they exist (e.g., 'art' or "another book")
    if (title_str.startswith("'") and title_str.endswith("'")) or \
       (title_str.startswith('"') and title_str.endswith('"')):
        title_str = title_str[1:-1] # Remove the outer quotes

    # List of common minor words that should not be capitalized in titles
    # unless they are the first word.
    minor_words = {
        "a", "an", "and", "as", "at", "but", "by", "for", "in", "nor", "of",
        "on", "or", "so", "the", "to", "up", "yet"
    }

    # Common contractions that should remain lowercase after an apostrophe
    contractions_after_apostrophe = {
        "'s", "'t", "'d", "'re", "'ll", "'ve", "'m"
    }

    # Helper function for consistent capitalization logic.
    def _capitalize_part(part: str) -> str:
        words = part.split()
        if not words:
            return ""

        formatted_words = []
        for i, word in enumerate(words):
            if not word: # Handle empty strings from multiple spaces
                continue

            formatted_word_candidate = word

            # Handle words like '#1Q84' - capitalize after the hash.
            if word.startswith('#') and len(word) > 1:
                formatted_word_candidate = '#' + word[1:].title()
            # Handle mixed alphanumeric words like '1Q84' - capitalize after the first digit.
            elif word[0].isdigit() and any(c.isalpha() for c in word):
                if len(word) > 1 and word[1].isalpha():
                    formatted_word_candidate = word[0] + word[1:].title()
                else:
                    formatted_word_candidate = word # For purely numeric words
            # Apply title case for most words, but lowercase minor words
            elif i == 0 or word.lower() not in minor_words:
                formatted_word_candidate = word.title()
                
                # Special handling for contractions after title case application
                for contraction in contractions_after_apostrophe:
                    if formatted_word_candidate.lower().endswith(contraction):
                        # Find the base word before the apostrophe and append the lowercase contraction
                        base_word = formatted_word_candidate.rsplit("'", 1)[0]
                        formatted_word_candidate = base_word + contraction.lower()
                        break # Assumes only one contraction per word
            else:
                formatted_word_candidate = word.lower()
            
            formatted_words.append(formatted_word_candidate)
        return ' '.join(formatted_words)

    # Split the title string into title and author based on the last comma.
    parts = title_str.rsplit(',', 1)

    if len(parts) == 2:
        # If successfully split into two parts (title, author).
        title_part = parts[0].strip()
        author_part = parts[1].strip()

        # Apply the helper capitalization to the title part.
        formatted_title = _capitalize_part(title_part)

        # Custom capitalization for author part, handling initials.
        formatted_author_parts = []
        for ap_word in author_part.split():
            if len(ap_word) == 1 and ap_word.isalpha(): # Single letter initial (e.g., J)
                formatted_author_parts.append(ap_word.upper())
            elif '.' in ap_word and len(ap_word) <= 2: # Initial with dot (e.g., J.)
                formatted_author_parts.append(ap_word.upper())
            else:
                formatted_author_parts.append(ap_word.title())
        formatted_author = ' '.join(formatted_author_parts)

        # Changed format from "{formatted_title}, {formatted_author}" to "{formatted_title}, by {formatted_author}"
        return f"{formatted_title} by {formatted_author}"
    else:
        # If no comma or only one part, assume it's just the title and apply capitalization.
        return _capitalize_part(title_str.strip())

# Create a display-friendly list with proper capitalization for the searchbox.
predetermined_book_list_display = sorted([
    format_book_title_for_display(book) for book in predetermined_book_list_backend_format
])


# Create a mapping from the display-friendly title to the original backend-formatted title.
# This is crucial for converting user selection back to the format your model expects.
display_to_backend_map = {
    format_book_title_for_display(book): book for book in predetermined_book_list_backend_format
}

# Function for st_searchbox to fetch suggestions
def search_books(search_term: str):
    if not search_term:
        results = predetermined_book_list_display[:50]
        return results
    search_term_lower = search_term.lower()
    filtered_results = [
        book for book in predetermined_book_list_display
        if search_term_lower in book.lower()
    ][:50] # Limit suggestions to 50 for performance
    return filtered_results


# --- Streamlit Application Layout ---

# Configure the Streamlit page settings.
st.set_page_config(
    page_title="Personalized Book Recommender",
    layout="centered", 
    initial_sidebar_state="auto" 
)

st.title("📚 Personalized Book Recommender")

st.markdown("""
Welcome to your personalized book recommendation engine!
To get started, select books you've read from the list along with your ratings (from 1 to 5).
The more books you add, the better the recommendations might be!
""")

st.header("Your Books & Ratings")

# Initialize session state for storing books and ratings if not already present
if 'user_books_data' not in st.session_state:
    st.session_state.user_books_data = [] # List of dictionaries: {'title': 'Book Title', 'rating': X}
# Initialize session state to store the last generated recommendations for feedback
if 'last_recommendations_display' not in st.session_state:
    st.session_state.last_recommendations_display = []
if 'last_recommendations_backend' not in st.session_state:
    st.session_state.last_recommendations_backend = []

# Initialize session state for feedback on recommended books
if 'feedback_ratings' not in st.session_state:
    st.session_state.feedback_ratings = {}

# New flag to control when recommendations are displayed
if 'display_recommendations' not in st.session_state:
    st.session_state.display_recommendations = False

# New flag to trigger recommendation generation when a rerun happens
if 'run_recommendation_logic_on_rerun' not in st.session_state:
    st.session_state.run_recommendation_logic_on_rerun = False


# Helper function to generate and display recommendations
def generate_and_display_recommendations_section(user_books, user_ratings, output_limit):
    with st.spinner("Generating recommendations..."):
        final_recommendations_df = get_combined_recommendations(
            user_books,
            user_ratings,
            genre_weight=FIXED_GENRE_WEIGHT,
            num_features=FIXED_NUM_FEATURES,
            new_user_regularization_strength=FIXED_NEW_USER_REGULARIZATION_STRENGTH
        )

    if final_recommendations_df is not None and not final_recommendations_df.empty:
        st.subheader("✨ Your Personalized Recommendations:")
        rec_list = recommend_by_multiplying_scores(final_recommendations_df, output_limit=output_limit)
        
        st.session_state.last_recommendations_display = [format_book_title_for_display(book) for book in rec_list]
        st.session_state.last_recommendations_backend = rec_list

        # Re-initialize feedback_ratings for the new recommendations
        st.session_state.feedback_ratings = {}
        for backend_title in st.session_state.last_recommendations_backend:
            st.session_state.feedback_ratings[backend_title] = {'read': False, 'rating': 3}

        if rec_list:
            for i, rec_book in enumerate(rec_list):
                st.write(f"- {format_book_title_for_display(rec_book)}")
        else:
            st.info("No recommendations found using this strategy. Try adding more books or different ratings!")
    else:
        st.error("Recommendation generation failed or returned an empty result. Please check your inputs and model functions.")


# --- Book Input Form ---

selected_book_display_title = st_searchbox(
    search_books,
    key="book_search_input",
    placeholder="Type to search for a book...",
    label="Select a Book You've Read" 
)

with st.form("book_input_form"):
    col1, col2 = st.columns([3, 1])
    with col2: 
        book_rating = st.slider("Your Rating (1-5)", 1, 5, 3)

    add_book_button = st.form_submit_button("Add Book")

    if add_book_button:
        if selected_book_display_title: # st_searchbox returns None if nothing is selected/typed
            # Convert the selected display title back to the backend format (lowercase)
            selected_book_backend_title = display_to_backend_map.get(selected_book_display_title)

            if selected_book_backend_title:
                # Check if the book has already been added to prevent duplicates
                if selected_book_backend_title not in [item['title'] for item in st.session_state.user_books_data]:
                    st.session_state.user_books_data.append({
                        'title': selected_book_backend_title, # Store backend-formatted title
                        'rating': book_rating
                    })
                    st.success(f"Added '{selected_book_display_title}' with rating {book_rating}!")
                    # Manually clear the searchbox by deleting its key from session state
                    if 'book_search_input' in st.session_state:
                        del st.session_state['book_search_input']
                else:
                    st.warning(f"'{selected_book_display_title}' has already been added.")
            else:
                st.error("Error: Could not find the selected book in the backend list. Please try again. Make sure you select from the suggestions.")
        else:
            st.warning("Please select a book from the list (type and choose from suggestions).")

# --- Function to remove a single book ---
def remove_single_book(book_backend_title_to_remove):
    # Filter out the book to be removed
    st.session_state.user_books_data = [
        item for item in st.session_state.user_books_data
        if item['title'] != book_backend_title_to_remove
    ]
    # Clear recommendations and feedback as input has changed
    st.session_state.last_recommendations_display = []
    st.session_state.last_recommendations_backend = []
    st.session_state.feedback_ratings = {}
    st.session_state.display_recommendations = False
    st.session_state.run_recommendation_logic_on_rerun = False
    st.rerun() # Force a rerun to update the UI

# --- Function to update a single book's rating ---
def update_single_book_rating(book_backend_title_to_update):
    # Retrieve the new rating from the session state key associated with the slider
    new_rating = st.session_state[f"user_book_rating_slider_{book_backend_title_to_update}"]
    for item in st.session_state.user_books_data:
        if item['title'] == book_backend_title_to_update:
            item['rating'] = new_rating
            break
    st.session_state.last_recommendations_display = []
    st.session_state.last_recommendations_backend = []
    st.session_state.feedback_ratings = {}
    st.session_state.display_recommendations = False # Hide old recommendations


# Display current list of books and a button to clear
if st.session_state.user_books_data:
    st.subheader("Books You've Added:")
    
    # Manually display books with a remove button and an adjustable rating slider for each
    for i, book_item in enumerate(st.session_state.user_books_data):
        col_title, col_rating_slider, col_remove = st.columns([0.5, 0.3, 0.2]) # Adjusted column widths
        with col_title:
            st.write(f"**{format_book_title_for_display(book_item['title'])}**")
        with col_rating_slider:
            # Add a slider to adjust the rating
            st.slider(
                "Rating",
                1, 5,
                value=book_item['rating'],
                key=f"user_book_rating_slider_{book_item['title']}", # Unique key for this slider
                on_change=update_single_book_rating,
                args=(book_item['title'],) # Pass only the title, new value is from key
            )
        with col_remove:
            st.button(
                "Remove",
                key=f"remove_book_{book_item['title']}_{i}", # Unique key for each button
                on_click=remove_single_book,
                args=(book_item['title'],)
            )

    st.markdown("---") # Separator between individual books and the "Clear All" button

    if st.button("Clear All Books"):
        st.session_state.user_books_data = []
        st.session_state.last_recommendations_display = [] # Clear recommendations as well
        st.session_state.last_recommendations_backend = [] # Clear recommendations as well
        st.session_state.feedback_ratings = {} # Clear feedback data as well
        st.session_state.display_recommendations = False # Hide recommendations
        st.session_state.run_recommendation_logic_on_rerun = False # Reset flag
        st.rerun() # Force a rerun


# --- Fixed Model Settings ---
FIXED_GENRE_WEIGHT = 0.8
FIXED_NUM_FEATURES = 600
FIXED_NEW_USER_REGULARIZATION_STRENGTH = 5

st.markdown("---")
st.header("Recommendation List Length") # Changed header to reflect only adjustable setting

output_limit = st.slider("Number of Recommendations to Display", 5, 50, 20)


# Recommendation Button
st.markdown("---")
st.header("Generate Recommendations")

if st.button("Get Recommendations", type="primary"):
    if st.session_state.user_books_data:
        st.session_state.display_recommendations = True
        st.session_state.run_recommendation_logic_on_rerun = True # Set flag to trigger generation
        st.rerun() # Force a rerun to display recommendations
    else:
        st.warning("Please add some books and ratings before generating recommendations!")


# --- Conditional Recommendation Display and Feedback ---
# This block will run on every rerun, and only execute the generation if the flag is set
if st.session_state.run_recommendation_logic_on_rerun and st.session_state.user_books_data:
    user_books = [item['title'] for item in st.session_state.user_books_data]
    user_ratings = [item['rating'] for item in st.session_state.user_books_data]
    generate_and_display_recommendations_section(user_books, user_ratings, output_limit)
    st.session_state.run_recommendation_logic_on_rerun = False # Reset flag after generation


# --- New Section: Feedback on Recommended Books ---
# This section only displays if there are recommendations to give feedback on
if st.session_state.display_recommendations and st.session_state.last_recommendations_display:
    st.markdown("---")
    st.header("Provide Feedback on Recommended Books")

    def update_feedback_read_status(backend_book_title):
        # Ensure the key exists before attempting to access it
        if backend_book_title not in st.session_state.feedback_ratings:
            st.session_state.feedback_ratings[backend_book_title] = {'read': False, 'rating': 3}
        
        current_status = st.session_state.get(f"read_checkbox_{backend_book_title}", False) # Get with default
        st.session_state.feedback_ratings[backend_book_title]['read'] = current_status
        # Reset rating to 0 if unchecked, or to default 3 if newly checked and was 0
        if not current_status:
            st.session_state.feedback_ratings[backend_book_title]['rating'] = 0
        elif st.session_state.feedback_ratings[backend_book_title]['rating'] == 0:
            st.session_state.feedback_ratings[backend_book_title]['rating'] = 3

    def update_feedback_rating(backend_book_title):
        # Ensure the key exists before attempting to access it
        if backend_book_title not in st.session_state.feedback_ratings:
            st.session_state.feedback_ratings[backend_book_title] = {'read': False, 'rating': 3}
            
        new_rating = st.session_state.get(f"rating_slider_{backend_book_title}", 3) # Get with default
        st.session_state.feedback_ratings[backend_book_title]['rating'] = new_rating

    st.markdown("Did you read one of our recommendations? Rate it here to get even better suggestions!")

    # Display feedback controls outside the form for immediate reactivity
    for i, rec_book_display_title in enumerate(st.session_state.last_recommendations_display):
        backend_book_title = st.session_state.last_recommendations_backend[i]
        
        # Ensure feedback_ratings is initialized for this book if it's a new recommendation
        if backend_book_title not in st.session_state.feedback_ratings:
            st.session_state.feedback_ratings[backend_book_title] = {'read': False, 'rating': 3}

        col_cb, col_rate = st.columns([0.7, 0.3])
        with col_cb:
            # Checkbox for "I have read this book"
            st.checkbox(
                f"I have read: **{rec_book_display_title}**",
                value=st.session_state.feedback_ratings[backend_book_title]['read'],
                key=f"read_checkbox_{backend_book_title}",
                on_change=update_feedback_read_status,
                args=(backend_book_title,)
            )
        with col_rate:
            # The rating slider is only displayed if the checkbox is marked as read
            if st.session_state.feedback_ratings[backend_book_title]['read']:
                st.slider(
                    "Your Rating",
                    1, 5,
                    value=st.session_state.feedback_ratings[backend_book_title]['rating'],
                    key=f"rating_slider_{backend_book_title}",
                    on_change=update_feedback_rating,
                    args=(backend_book_title,)
                )
    
    # Form for the submit button only
    with st.form("feedback_submit_form", clear_on_submit=True):
        add_feedback_button = st.form_submit_button("Submit Feedback & Get New Recommendations")

        if add_feedback_button:
            feedback_processed_count = 0
            for backend_title, data in st.session_state.feedback_ratings.items():
                if data['read'] and data['rating'] > 0: # Only process if read and rated
                    found_and_updated = False
                    for item in st.session_state.user_books_data:
                        if item['title'] == backend_title:
                            item['rating'] = data['rating']
                            st.success(f"Updated rating for '{format_book_title_for_display(backend_title)}' to {data['rating']}!")
                            found_and_updated = True
                            break
                    if not found_and_updated:
                        st.session_state.user_books_data.append({
                            'title': backend_title,
                            'rating': data['rating']
                        })
                        st.success(f"Added feedback for '{format_book_title_for_display(backend_title)}' with rating {data['rating']}!")
                    feedback_processed_count += 1
            
            if feedback_processed_count == 0:
                st.info("No new feedback was provided for recommended books.")
            
            # Set the flag to trigger a new recommendation generation on the next rerun
            st.session_state.run_recommendation_logic_on_rerun = True
            st.rerun() # Force a rerun to apply feedback and get new recommendations

else:
    # Only show this message if recommendations haven't been displayed yet
    if not st.session_state.display_recommendations:
        st.info("Generate recommendations first to provide feedback on them.")
