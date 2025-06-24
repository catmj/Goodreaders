import streamlit as st
import pandas as pd
import os # Import the os module to handle file paths

# --- IMPORT RECOMMENDATION FUNCTIONS FROM YOUR LOCAL FILE ---
# Ensure the 'combined.py' file is in the specified path relative to your Streamlit app.
try:
    # Attempt to import the necessary functions from the 'combined' module within 'combined_model' package.
    # This assumes 'combined_model' is a directory in the parent directory of this script,
    # and 'combined.py' is inside 'combined_model'.
    from combined_model.combined import (
        get_combined_recommendations,
        recommend_by_multiplying_scores, # Only importing this specific recommendation strategy
    )
    st.success("Successfully loaded recommendation functions from 'combined.py'!")
except ImportError as e:
    # If the import fails, display an error message and stop the Streamlit application.
    st.error(f"Error loading recommendation functions: {e}. Please ensure 'combined.py' is in the correct path ('../combined_model/') and all its dependencies are installed.")
    st.stop() # Stop the app if functions cannot be loaded


# --- Predetermined List of Books (Original Lowercase Format for Backend) ---
# Define the relative path to the book list file.
book_list_file_path = "../collab_filtering/trained_data/book_list_features_600_lambda_5.txt"
predetermined_book_list_backend_format = []

try:
    # Get the directory of the current script (app.py) to construct an absolute path.
    script_dir = os.path.dirname(__file__)
    # Construct the absolute path to the book list file, making it robust to different execution contexts.
    abs_file_path = os.path.join(script_dir, book_list_file_path)

    # Open and read the book list file, adding each line (stripped of whitespace) to the list.
    with open(abs_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            predetermined_book_list_backend_format.append(line.strip()) # Read each line and strip whitespace
    st.success(f"Successfully loaded {len(predetermined_book_list_backend_format)} books from '{book_list_file_path}'!")
except FileNotFoundError:
    # Handle the case where the book list file does not exist.
    st.error(f"Error: The book list file '{book_list_file_path}' was not found. Please check the path and ensure the file exists.")
    st.stop() # Stop the app if the book list cannot be loaded
except Exception as e:
    # Handle any other exceptions that might occur during file loading.
    st.error(f"An error occurred while loading the book list: {e}")
    st.stop()


# Function to format book titles for display in the Streamlit UI.
# This function handles capitalization, quotes, and author/title separation.
def format_book_title_for_display(title_str: str) -> str:
    # First, handle outer quotes if they exist (e.g., 'art' or "another book")
    if (title_str.startswith("'") and title_str.endswith("'")) or \
       (title_str.startswith('"') and title_str.endswith('"')):
        title_str = title_str[1:-1] # Remove the outer quotes

    # Helper function for consistent capitalization logic.
    def _capitalize_part(part: str) -> str:
        formatted_words = []
        for word in part.split():
            if word.startswith('#') and len(word) > 1:
                # Handle words like '#1Q84' - capitalize after the hash.
                formatted_words.append('#' + word[1:].title())
            elif word and word[0].isdigit() and any(c.isalpha() for c in word):
                # Handle mixed alphanumeric words like '1Q84' - capitalize after the first digit.
                if len(word) > 1 and word[1].isalpha():
                    formatted_words.append(word[0] + word[1:].title())
                else:
                    # For purely numeric words or single digits, no title case.
                    formatted_words.append(word)
            else:
                # Default to title case for most words.
                formatted_words.append(word.title())
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
        return f"{formatted_title}, by {formatted_author}"
    else:
        # If no comma or only one part, assume it's just the title and apply capitalization.
        return _capitalize_part(title_str.strip())

# Create a display-friendly list with proper capitalization for the selectbox.
predetermined_book_list_display = sorted([
    format_book_title_for_display(book) for book in predetermined_book_list_backend_format
])

# Create a mapping from the display-friendly title to the original backend-formatted title.
# This is crucial for converting user selection back to the format your model expects.
display_to_backend_map = {
    format_book_title_for_display(book): book for book in predetermined_book_list_backend_format
}


# --- Streamlit Application Layout ---

# Configure the Streamlit page settings.
st.set_page_config(
    page_title="Personalized Book Recommender",
    layout="centered", # Layout can be 'centered' or 'wide'.
    initial_sidebar_state="auto" # 'auto', 'expanded', or 'collapsed'.
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


# Input form for adding a new book
with st.form("book_input_form", clear_on_submit=True):
    col1, col2 = st.columns([3, 1])
    with col1:
        # Use st.selectbox for predetermined list with display-friendly titles
        selected_book_display_title = st.selectbox(
            "Select a Book You've Read",
            options=["-- Select a book --"] + predetermined_book_list_display, # Use display list
            help="Choose a book from the available list."
        )
    with col2:
        book_rating = st.slider("Your Rating (1-5)", 1, 5, 3)

    add_book_button = st.form_submit_button("Add Book")

    if add_book_button:
        if selected_book_display_title and selected_book_display_title != "-- Select a book --":
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
                else:
                    st.warning(f"'{selected_book_display_title}' has already been added.")
            else:
                st.error("Error: Could not find the selected book in the backend list. Please try again.")
        else:
            st.warning("Please select a book from the list.")

# Display current list of books and a button to clear
if st.session_state.user_books_data:
    st.subheader("Books You've Added:")
    df_books = pd.DataFrame(st.session_state.user_books_data)
    # Display a more user-friendly title (capitalized)
    df_books_display = df_books.copy()
    df_books_display['Book Title'] = df_books_display['title'].apply(lambda x: format_book_title_for_display(x))
    df_books_display['Rating'] = df_books_display['rating']
    st.dataframe(df_books_display[['Book Title', 'Rating']], hide_index=True, use_container_width=True)

    if st.button("Clear All Books"):
        st.session_state.user_books_data = []
        st.session_state.last_recommendations_display = [] # Clear recommendations as well
        st.session_state.last_recommendations_backend = [] # Clear recommendations as well
        st.rerun() # Rerun to clear the displayed dataframe immediately

# --- Fixed Model Settings (Backend Configuration) ---
# These values are now fixed in the backend as per your request.
# Based on previous analysis, nf600_rs5_gw0.8 seemed to perform best.
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
        # Prepare data for your combined recommendation function (using backend-formatted titles)
        user_books = [item['title'] for item in st.session_state.user_books_data]
        user_ratings = [item['rating'] for item in st.session_state.user_books_data]

        with st.spinner("Generating recommendations..."):
            # Call your actual `get_combined_recommendations` function with fixed settings
            final_recommendations_df = get_combined_recommendations(
                user_books,
                user_ratings,
                genre_weight=FIXED_GENRE_WEIGHT,
                num_features=FIXED_NUM_FEATURES,
                new_user_regularization_strength=FIXED_NEW_USER_REGULARIZATION_STRENGTH
            )

        if final_recommendations_df is not None and not final_recommendations_df.empty:
            st.subheader("✨ Your Personalized Recommendations:")

            # Only Strategy: Scores Multiplied Together
            st.markdown("#### Recommendations (Hybrid: Content-Based & Collaborative Filtering Scores Multiplied)")
            rec_list = recommend_by_multiplying_scores(final_recommendations_df, output_limit=output_limit)
            
            # Store the current recommendations in session state for feedback functionality
            st.session_state.last_recommendations_display = [format_book_title_for_display(book) for book in rec_list]
            st.session_state.last_recommendations_backend = rec_list

            if rec_list:
                for i, rec_book in enumerate(rec_list):
                    st.write(f"- {format_book_title_for_display(rec_book)}") # Display with capitalized titles
            else:
                st.info("No recommendations found using this strategy. Try adding more books or different ratings!")

        else:
            st.error("Recommendation generation failed or returned an empty result. Please check your inputs and model functions.")
    else:
        st.warning("Please add some books and ratings before generating recommendations!")

# --- New Section: Feedback on Recommended Books ---
st.markdown("---")
st.header("Provide Feedback on Recommended Books")

if st.session_state.last_recommendations_display:
    st.markdown("Did you read one of our recommendations? Rate it here to get even better suggestions!")
    with st.form("feedback_form", clear_on_submit=True):
        col1_feedback, col2_feedback = st.columns([3, 1])
        with col1_feedback:
            # Selectbox for the user to choose from the *last generated* recommendations
            selected_feedback_book_display_title = st.selectbox(
                "Select a Recommended Book to Rate",
                options=["-- Select a book --"] + st.session_state.last_recommendations_display,
                help="Choose a book from the last recommendations generated."
            )
        with col2_feedback:
            feedback_rating = st.slider("Your Rating (1-5)", 1, 5, 3, key="feedback_rating_slider")

        add_feedback_button = st.form_submit_button("Add Feedback & Regenerate Recommendations")

        if add_feedback_button:
            if selected_feedback_book_display_title and selected_feedback_book_display_title != "-- Select a book --":
                # Convert the selected display title back to the backend format
                selected_feedback_book_backend_title = display_to_backend_map.get(selected_feedback_book_display_title)

                if selected_feedback_book_backend_title:
                    # Check if the book has already been added to prevent duplicates
                    if selected_feedback_book_backend_title not in [item['title'] for item in st.session_state.user_books_data]:
                        st.session_state.user_books_data.append({
                            'title': selected_feedback_book_backend_title,
                            'rating': feedback_rating
                        })
                        st.success(f"Added feedback for '{selected_feedback_book_display_title}' with rating {feedback_rating}!")
                        # Clear previous recommendations so that the next run generates fresh ones
                        st.session_state.last_recommendations_display = []
                        st.session_state.last_recommendations_backend = []
                        st.rerun() # Rerun to trigger new recommendations with updated user data
                    else:
                        st.warning(f"'{selected_feedback_book_display_title}' has already been added. Its rating can be adjusted in 'Your Books & Ratings' section.")
                else:
                    st.error("Error: Could not find the selected book for feedback. Please try again.")
            else:
                st.warning("Please select a book from the recommended list to provide feedback.")
else:
    st.info("Generate recommendations first to provide feedback on them.")


st.markdown("""
---
**Note:** The recommendation functions are imported from `combined_model/combined.py`. Model parameters (Genre Weight, Number of Features, User Regularization Strength) are now fixed in the backend for consistent performance.
""")