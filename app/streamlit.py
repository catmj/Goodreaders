import streamlit as st
import pandas as pd

# --- IMPORT RECOMMENDATION FUNCTIONS FROM YOUR LOCAL FILE ---
# Ensure the 'combined.py' file is in the specified path relative to your Streamlit app.
try:
    from combined_model.combined import (
        get_combined_recommendations,
        recommend_by_multiplying_scores, # Only importing this specific recommendation strategy
    )
    st.success("Successfully loaded recommendation functions from 'combined.py'!")
except ImportError as e:
    st.error(f"Error loading recommendation functions: {e}. Please ensure 'combined.py' is in the correct path ('../combined_model/') and all its dependencies are installed.")
    st.stop() # Stop the app if functions cannot be loaded


# --- Predetermined List of Books (Original Lowercase Format for Backend) ---
# You should replace this with your actual, comprehensive list of books
# that your recommendation model understands.
# The format should match what your backend expects (e.g., "title, author")
predetermined_book_list_backend_format = [
    "the hobbit, j.r.r. tolkien",
    "the lord of the rings, j.r.r. tolkien",
    "gideon the ninth, tamsyn muir",
    "the fifth season, n.k. jemisin",
    "iron widow, xiran jay zhao",
    "all systems red, martha wells",
    "witch king, martha wells",
    "deathless, catherynne m. valente",
    "the adventures of amina al-sirafi, shannon chakraborty",
    "the city of brass, s.a. chakraborty",
    "flight of magpies, k.j. charles",
    "ninefox gambit, yoon ha lee",
    "she who became the sun, shelley parker-chan",
    "boyfriend material, alexis hall",
    "the traitor baru cormorant, seth dickinson",
    "a memory called empire, arkady martine",
    "this is how you lose the time war, amal el-mohtar",
    "summer sons, lee mandelo",
    "1984, george orwell",
    "brave new world, aldous huxley",
    "fahrenheit 451, ray bradbury",
    "the handmaid's tale, margaret atwood",
    "dune, frank herbert",
    "the martian, andy weir",
    "project hail mary, andy weir",
    "circe, madeline miller",
    "the song of achilles, madeline miller",
    "the silent patient, alex michaelides",
    "where the crawdads sing, delia owens",
    "the midnight library, matt haig",
    "the guest list, lucy fokley",
    "station eleven, emily st. john mandel",
    "the name of the wind, patrick rothfuss",
    "mistborn: the final empire, brandon sanderson",
    "the priory of the orange tree, samantha shannon",
    "a wizard of earthsea, ursula k. le guin",
    "hyperion, dan simmons",
    "neuromancer, william gibson",
    "foundation, isaac asimov",
    "a fire upon the deep, vernor vinge",
    "the chronicles of narnia, c.s. lewis",
    "percy jackson & the lightning thief, rick riordan",
    "eragon, christopher paolini",
    "the poppy war, r.f. kuang",
    "babel, r.f. kuang",
    "red rising, pierce brown",
    "golden son, pierce brown",
    "morning star, pierce brown",
    "light bringer, pierce brown",
    "the curious case of benjamin button, f. scott fitzgerald", # Added for testing comma in title
    "the book, with a comma in title, author name" # Added for testing comma in title
]

# Create a display-friendly list with proper capitalization for the selectbox
def format_book_title_for_display(title_str: str) -> str:
    parts = title_str.rsplit(',', 1) # Split only on the last comma
    if len(parts) == 2:
        title_part = parts[0].strip().title()
        author_part = parts[1].strip()
        # Special handling for initials in author name (e.g., J.R.R.)
        formatted_author_parts = []
        for ap_word in author_part.split():
            if len(ap_word) == 1 and ap_word.isalpha(): # Single letter initial
                formatted_author_parts.append(ap_word.upper())
            elif '.' in ap_word and len(ap_word) <= 2: # Initial with dot, e.g., J.
                formatted_author_parts.append(ap_word.upper())
            else:
                formatted_author_parts.append(ap_word.title())
        formatted_author = ' '.join(formatted_author_parts)
        return f"{title_part}, {formatted_author}"
    else:
        # If no comma or only one part, assume it's just the title
        return title_str.strip().title()

predetermined_book_list_display = sorted([
    format_book_title_for_display(book) for book in predetermined_book_list_backend_format
])

# Create a mapping from display title to backend title for easy lookup
# This is crucial for converting user selection back to the format your model expects.
display_to_backend_map = {
    format_book_title_for_display(book): book for book in predetermined_book_list_backend_format
}


# --- Streamlit Application Layout ---

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
            if rec_list:
                for i, rec_book in enumerate(rec_list):
                    st.write(f"- {format_book_title_for_display(rec_book)}") # Display with capitalized titles
            else:
                st.info("No recommendations found using this strategy. Try adding more books or different ratings!")

        else:
            st.error("Recommendation generation failed or returned an empty result. Please check your inputs and model functions.")
    else:
        st.warning("Please add some books and ratings before generating recommendations!")

st.markdown("""
---
**Note:** The recommendation functions are imported from `combined_model/combined.py`. Model parameters (Genre Weight, Number of Features, User Regularization Strength) are now fixed in the backend for consistent performance.
""")
