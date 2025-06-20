import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

# --- Configuration ---
DATA_PATH = '../fake_data/' # This path is for the narrowed_recommendations.csv
PRETRAINED_DATA_PATH = './trained_data/' # Path where X_features_XXX.npy, book_list_features_XXX.txt, and Y_mean_features_XXX.npy are saved
NARROWED_RECOMMENDATIONS_FILE = DATA_PATH + 'narrowed_recommendations.csv'
RECOMMENDATION_OUTPUT_FOLDER = './recommendations_output/' # Folder to save recommendation lists
RECOMMENDATION_OUTPUT_FILENAME = 'new_user_recommendations.txt' # Default filename for recommendations
NUM_FEATURES = 300 # This must match the NUM_FEATURES used when saving X and book_list
TOP_N_RECOMMENDATIONS = 30 # Number of recommendations to show

os.makedirs(RECOMMENDATION_OUTPUT_FOLDER, exist_ok=True)

# --- Data Loading for Narrowed Recommendations ---
def load_narrowed_recommendations(file_path):
    """
    Loads a list of book titles for narrowed recommendations from a CSV file.
    Assumes the CSV has a column named 'book'.
    
    Args:
        file_path (str): Path to the narrowed recommendations CSV file.

    Returns:
        list: A list of book titles. Returns an empty list if file not found or empty.
    """
    if not os.path.exists(file_path):
        print(f"Info: Narrowed recommendations file '{file_path}' not found. Will use full book list if no recommendations are provided.")
        return []
        
    try:
        df = pd.read_csv(file_path)
        if 'book' in df.columns:
            if not df['book'].empty:
                print(f"Loaded {len(df['book'])} books from narrowed recommendations file: {file_path}")
                return df['book'].tolist()
            else:
                print(f"Info: Narrowed recommendations file '{file_path}' is empty. Will use full book list.")
                return []
        else:
            print(f"Warning: '{file_path}' does not contain a 'book' column. Will use full book list.")
            return []
    except pd.errors.EmptyDataError:
        print(f"Info: Narrowed recommendations file '{file_path}' is empty. Will use full book list.")
        return []
    except Exception as e:
        print(f"Error loading narrowed recommendations from '{file_path}': {e}. Will use full book list.")
        return []

# --- Loading Pre-trained Data (X, book_list, and Y_mean) ---
def load_pre_trained_data(num_features, pretrained_data_path):
    """
    Loads pre-trained book features (X), book list, and Y_mean from specified files.

    Args:
        num_features (int): Number of features per book (used in filename).
        pretrained_data_path (str): The directory where the .npy and .txt files are saved.

    Returns:
        tuple: (X, book_list, Y_mean)
               X (ndarray): Matrix of book features.
               book_list (list): List of book titles.
               Y_mean (ndarray): Mean rating for each book.
               Returns (None, None, None) if files not found.
    """
    x_filename = os.path.join(pretrained_data_path, f'X_features_{num_features}.npy')
    book_list_filename = os.path.join(pretrained_data_path, f'book_list_features_{num_features}.txt')
    y_mean_filename = os.path.join(pretrained_data_path, f'Y_mean_features_{num_features}.npy')

    X = None
    book_list = None
    Y_mean = None

    try:
        X = np.load(x_filename)
        print(f"Loaded X matrix from: {x_filename}")
    except FileNotFoundError:
        print(f"Error: X matrix file '{x_filename}' not found. Please ensure it exists.")
        return None, None, None
    
    try:
        with open(book_list_filename, 'r', encoding='utf-8') as f:
            book_list = [line.strip() for line in f]
        print(f"Loaded book list from: {book_list_filename}")
    except FileNotFoundError:
        print(f"Error: Book list file '{book_list_filename}' not found. Please ensure it exists.")
        return None, None, None

    try:
        Y_mean = np.load(y_mean_filename)
        print(f"Loaded Y_mean from: {y_mean_filename}")
    except FileNotFoundError:
        print(f"Error: Y_mean file '{y_mean_filename}' not found. Please ensure it exists.")
        return None, None, None
    
    return X, book_list, Y_mean

# --- Prepare User Ratings ---
def prepare_user_ratings(my_books, my_ratings, book_list, num_books):
    """
    Prepares a test user's ratings array based on their provided books and ratings.

    Args:
        my_books (list): List of book titles rated by the user.
        my_ratings (list): List of ratings corresponding to my_books.
        book_list (list): The master list of all book titles.
        num_books (int): Total number of books.

    Returns:
        tuple: (test_ratings, test_rated_indices)
               test_ratings (ndarray): An array of the user's ratings, with NaN for unrated books.
               test_rated_indices (list): Indices of books the user has rated.
    """
    test_ratings = np.full(num_books, np.nan)
    test_rated_indices = []

    for book_title, rating in zip(my_books, my_ratings):
        try:
            idx = book_list.index(book_title)
            test_ratings[idx] = rating
            test_rated_indices.append(idx)
        except ValueError:
            print(f"Warning: Book '{book_title}' not found in the book list. Skipping.")
            continue
    
    return test_ratings, test_rated_indices

# --- Train Linear Regression Model ---
def train_linear_regression_model(X, Y_mean, test_ratings):
    """
    Trains a Linear Regression model for a specific user to learn their preferences
    based on their ratings and book features.

    Args:
        X (ndarray): Matrix of book features.
        Y_mean (ndarray): Mean rating for each book.
        test_ratings (ndarray): Array of the user's ratings.

    Returns:
        tuple: (W, b)
               W (ndarray): Learned weights (coefficients) for the user.
               b (float): Learned bias (intercept) for the user.
    """
    # Calculate user's ratings relative to the mean
    y_user = test_ratings - Y_mean
    
    # Create a mask for rated books (where y_user is not NaN)
    rated_mask = ~np.isnan(y_user)
    
    # Select features and target for rated books
    X_rated, y_rated = X[rated_mask], y_user[rated_mask]
    
    # Train the linear regression model
    reg = LinearRegression().fit(X_rated, y_rated)
    
    W = reg.coef_
    b = reg.intercept_
    
    return W, b

# --- Function to generate user predictions ---
def generate_user_predictions(X_all, Y_mean_all, book_list_final_all, num_books_all, my_books, my_ratings):
    """
    Prepares user ratings, trains the user's linear regression model,
    and calculates predicted ratings for all books.

    Args:
        X_all (ndarray): The full matrix of all book features.
        Y_mean_all (ndarray): The mean rating for all books.
        book_list_final_all (list): The master list of all book titles.
        num_books_all (int): Total number of books.
        my_books (list): List of book titles rated by the user.
        my_ratings (list): List of ratings corresponding to my_books.

    Returns:
        tuple: (user_predictions_full, test_rated_indices_all, W_user, b_user)
               user_predictions_full (ndarray): Predicted ratings for all books.
               test_rated_indices_all (list): Indices of books the user has already rated.
               W_user (ndarray): Learned weights for the user.
               b_user (float): Learned bias for the user.
    """
    test_ratings_all, test_rated_indices_all = prepare_user_ratings(
        my_books, my_ratings, book_list_final_all, num_books_all
    )

    print('\n--- Your Provided Ratings (against full book list) ---')
    if test_rated_indices_all:
        for idx in test_rated_indices_all:
            print(f'Rated {test_ratings_all[idx]} for: {book_list_final_all[idx]}')
    else:
        print("No provided books were found in the main book list for training.")

    W_user, b_user = train_linear_regression_model(X_all, Y_mean_all, test_ratings_all)
    print("User-specific linear regression model trained.")

    # Calculate predictions for all books using the trained user model
    user_predictions_full_norm = np.matmul(X_all, W_user) + b_user
    user_predictions_full = user_predictions_full_norm + Y_mean_all
    
    return user_predictions_full, test_rated_indices_all, W_user, b_user

# --- Function to get target recommendation data ---
def get_target_recommendation_data(
    narrowed_recommendations_file, X_all, Y_mean_all, book_list_final_all
):
    """
    Determines whether to use a narrowed list or the full book list for recommendations.

    Args:
        narrowed_recommendations_file (str): Path to the narrowed recommendations CSV.
        X_all (ndarray): The full matrix of all book features.
        Y_mean_all (ndarray): The mean rating for all books.
        book_list_final_all (list): The master list of all book titles.

    Returns:
        tuple: (X_target, Y_mean_target, book_list_target, is_narrowed)
               X_target (ndarray): Feature matrix for the chosen target list.
               Y_mean_target (ndarray): Mean ratings for the chosen target list.
               book_list_target (list): Book titles for the chosen target list.
               is_narrowed (bool): True if a narrowed list is being used, False otherwise.
    """
    narrowed_book_names = load_narrowed_recommendations(narrowed_recommendations_file)
    
    X_target = X_all
    Y_mean_target = Y_mean_all
    book_list_target = book_list_final_all
    is_narrowed = False

    if narrowed_book_names:
        book_to_idx_map_all = {book_name: i for i, book_name in enumerate(book_list_final_all)}
        
        filtered_indices = []
        filtered_book_list = []
        for book_name in narrowed_book_names:
            idx = book_to_idx_map_all.get(book_name)
            if idx is not None:
                filtered_indices.append(idx)
                filtered_book_list.append(book_name)
            else:
                print(f"Warning: Narrowed book '{book_name}' not found in main book list. Skipping.")

        if filtered_indices:
            X_target = X_all[filtered_indices, :]
            Y_mean_target = Y_mean_all[filtered_indices]
            book_list_target = filtered_book_list
            is_narrowed = True
        else:
            print("No matching books found for narrowed recommendations. Defaulting to full book list.")
    
    return X_target, Y_mean_target, book_list_target, is_narrowed

# --- Display and Get Top Recommendations ---
def display_and_get_top_recommendations(
    user_predictions_target, book_list_target, original_book_list_all, test_rated_indices_all, num_recommendations,
    my_books_original, my_ratings_original, is_narrowed_list=False
):
    """
    Displays top N recommendations and rated books for a given set of predictions,
    and returns the list of top recommendations.

    Args:
        user_predictions_target (ndarray): Predicted ratings for the target list of books.
        book_list_target (list): List of book titles from the target recommendations.
        original_book_list_all (list): The full master list of all book titles.
        test_rated_indices_all (list): Indices of books the user has already rated in the ALL book list.
        num_recommendations (int): Number of top recommendations to display.
        my_books_original (list): The user's original list of rated book titles.
        my_ratings_original (list): The user's original list of ratings.
        is_narrowed_list (bool): True if book_list_target is a narrowed list, False if it's the full list.

    Returns:
        list: A list of tuples, each containing (predicted_rating, book_title) for the top unrated recommendations.
    """
    # Create a mapping from full book list to original index for quick lookup
    full_book_to_idx = {book_name: i for i, book_name in enumerate(original_book_list_all)}
    
    # Identify which books in the target list the user has ALREADY rated
    target_rated_indices = []
    for book_name in book_list_target:
        original_idx = full_book_to_idx.get(book_name)
        if original_idx is not None and original_idx in test_rated_indices_all:
            target_rated_indices.append(book_list_target.index(book_name))

    # Sort predictions by rating in descending order
    sorted_indices = np.argsort(user_predictions_target)[::-1]

    list_type_label = "Narrowed List" if is_narrowed_list else "Full List"
    print(f"\n--- Top {num_recommendations} Recommendations from {list_type_label} ---")
    
    top_recommendations_for_export = []
    recommended_count = 0
    for idx in sorted_indices:
        if idx not in target_rated_indices: # Only recommend books the user hasn't rated
            predicted_rating = user_predictions_target[idx]
            book_title = book_list_target[idx]
            print(f"Predicting rating {predicted_rating:0.2f} for book: {book_title}")
            top_recommendations_for_export.append((predicted_rating, book_title))
            recommended_count += 1
            if recommended_count >= num_recommendations:
                break
    
    print(f"\n--- Original vs. Predicted Ratings for Books You've Rated (from {list_type_label}) ---")
    found_rated_in_target = False
    for i, book_name_target in enumerate(book_list_target):
        if book_name_target in my_books_original:
            original_rating_idx = my_books_original.index(book_name_target)
            print(f"Book: {book_name_target}, Original Rating: {my_ratings_original[original_rating_idx]}, Predicted Rating: {user_predictions_target[i]:0.2f}")
            found_rated_in_target = True
    if not found_rated_in_target:
        print(f"No user-rated books were found within the {list_type_label} for comparison.")

    return top_recommendations_for_export

# --- Function to export recommendation list ---
def export_recommendation_list(recommendation_list, output_filename, output_folder):
    """
    Exports a list of recommendations (predicted_rating, book_title) to a text file.

    Args:
        recommendation_list (list): A list of tuples, e.g., [(rating, title), ...].
        output_filename (str): The name of the output file.
        output_folder (str): The folder where the file will be saved.
    """
    os.makedirs(output_folder, exist_ok=True) # Ensure the output folder exists
    filepath = os.path.join(output_folder, output_filename)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("--- Your Top Recommendations ---\n\n")
            if not recommendation_list:
                f.write("No recommendations generated.\n")
            else:
                for rating, book_title in recommendation_list:
                    f.write(f"Predicted Rating: {rating:0.2f}, Book: {book_title}\n")
        print(f"\nRecommendations successfully exported to: {filepath}")
    except Exception as e:
        print(f"Error exporting recommendations to {filepath}: {e}")


## Core Recommendation Function

def get_recommendations_for_user(
    my_books: list,
    my_ratings: list,
    num_features: int = NUM_FEATURES,
    pretrained_data_path: str = PRETRAINED_DATA_PATH,
    narrowed_recommendations_file: str = NARROWED_RECOMMENDATIONS_FILE,
    recommendation_output_folder: str = RECOMMENDATION_OUTPUT_FOLDER,
    recommendation_output_filename: str = RECOMMENDATION_OUTPUT_FILENAME,
    top_n_recommendations: int = TOP_N_RECOMMENDATIONS
) -> list:
    """
    Generates and exports book recommendations for a new user based on their provided ratings.

    This function encapsulates the entire recommendation pipeline:
    1. Loads pre-trained collaborative filtering model components (X, Y_mean, book_list).
    2. Prepares the new user's ratings and trains a personal preference model.
    3. Determines if recommendations should be narrowed by a provided list or use the full catalog.
    4. Calculates and displays the top N recommendations.
    5. Exports the top N recommendations to a text file.

    Args:
        my_books (list): A list of book titles rated by the user.
        my_ratings (list): A list of ratings (integers) corresponding to `my_books`.
        num_features (int): Number of features used in the pre-trained model.
        pretrained_data_path (str): Path to the folder containing pre-trained X, Y_mean, and book_list.
        narrowed_recommendations_file (str): Path to the CSV file for narrowed recommendations.
                                             If not found or empty, the full book list is used.
        recommendation_output_folder (str): Folder where the recommendation list will be saved.
        recommendation_output_filename (str): Name of the file to save the recommendations.
        top_n_recommendations (int): The number of top recommendations to display and export.

    Returns:
        list: A list of tuples, each containing (predicted_rating, book_title) for the top unrated recommendations.
              Returns an empty list if no recommendations could be generated.
    """
    print("\n--- Starting Recommendation Generation for New User ---")

    # 1. Load the pre-trained X, book_list, and Y_mean
    X_all, book_list_final_all, Y_mean_all = load_pre_trained_data(num_features, pretrained_data_path)
    
    if X_all is None or book_list_final_all is None or Y_mean_all is None:
        print("Error: Could not load all pre-trained data. Cannot generate recommendations.")
        return []

    num_books_all = X_all.shape[0]

    # 2. Generate predictions for the new user across all books
    user_predictions_full, test_rated_indices_all, W_user, b_user = \
        generate_user_predictions(X_all, Y_mean_all, book_list_final_all, num_books_all, my_books, my_ratings)
    
    if W_user is None or b_user is None:
        print("Error: Could not train user model. Cannot generate recommendations.")
        return []

    # 3. Determine target recommendation data (narrowed or full)
    X_target, Y_mean_target, book_list_target, is_narrowed = get_target_recommendation_data(
        narrowed_recommendations_file, X_all, Y_mean_all, book_list_final_all
    )
    
    # Calculate predictions specifically for the target list (narrowed or full)
    # We use the W_user and b_user trained on the full feature space
    user_predictions_target_norm = np.matmul(X_target, W_user) + b_user
    user_predictions_target = user_predictions_target_norm + Y_mean_target

    # 4. Display and get the top recommendations
    top_recommendations_list = display_and_get_top_recommendations(
        user_predictions_target, book_list_target, book_list_final_all, test_rated_indices_all, top_n_recommendations,
        my_books, my_ratings, is_narrowed_list=is_narrowed
    )

    # 5. Export the recommendation list
    export_recommendation_list(top_recommendations_list, recommendation_output_filename, recommendation_output_folder)
    
    print("\n--- Recommendation Generation Complete ---")
    return top_recommendations_list




if __name__ == "__main__":
    # Define a sample new user's ratings
    sample_my_books = [
        "gideon the ninth, tamsyn muir", 
        "the fifth season, n.k. jemisin", 
        "the king of attolia, megan whalen turner", 
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
        "summer sons, lee mandelo"

    ]
    sample_my_ratings = [5, 5, 5, 3, 4, 5, 2, 4, 5, 4, 5, 4, 4, 5, 4, 5, 4]

    # Call the new master function to get recommendations for the sample user
    final_recommendations = get_recommendations_for_user(
        my_books=sample_my_books,
        my_ratings=sample_my_ratings
    )

    if final_recommendations:
        print(f"\nGenerated {len(final_recommendations)} recommendations.")
    else:
        print("\nNo recommendations were generated.")

    # You could now use final_recommendations for other purposes in your application