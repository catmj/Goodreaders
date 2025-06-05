import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- Configuration ---
DATA_PATH = '../fake_data/'
RATINGS_FILE = DATA_PATH + 'ratings.csv'
# If you were loading X and book_list from files, you'd define their paths here:
# X_FILE = DATA_PATH + 'book_features.npy'
# BOOK_LIST_FILE = DATA_PATH + 'book_list.txt'
NUM_FEATURES = 200
TOP_N_RECOMMENDATIONS = 10 # Number of recommendations to show

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(ratings_file):
    """
    Loads and preprocesses the ratings data, returning necessary matrices and lists.
    
    Args:
        ratings_file (str): Path to the ratings CSV file.

    Returns:
        tuple: (Y, Y_norm, Y_mean, book_list, user_list)
               Y (ndarray): Original ratings matrix (with NaNs).
               Y_norm (ndarray): Mean-normalized ratings matrix (NaNs replaced with 0).
               Y_mean (ndarray): Mean rating for each book.
               book_list (list): List of book titles.
               user_list (list): List of user IDs.
               Returns (None, None, None, None, None) if file not found.
    """
    try:
        ratings = pd.read_csv(ratings_file)
    except FileNotFoundError:
        print(f"Error: {ratings_file} not found. Please check the path.")
        return None, None, None, None, None

    ratings["book"] = ratings[['Title', 'Author']].agg(', '.join, axis=1)
    ratings = ratings[["book", "User_id", "Rating"]]

    # Filter invalid ratings and convert to integer
    ratings = ratings[ratings['Rating'] != 'Invalid rating'].copy()
    ratings['Rating'] = ratings['Rating'].astype(int)

    pivot_ratings = ratings.pivot_table(index='book', columns='User_id', values='Rating')

    Y = pivot_ratings.to_numpy()
    
    # Calculate mean and normalized Y
    Y_mean = np.nanmean(Y, axis=1)
    Y_norm = (Y.T - Y_mean).T
    Y_norm = np.nan_to_num(Y_norm) # Replace NaNs with 0 in the normalized matrix for computation

    book_list = pivot_ratings.index.tolist()
    user_list = pivot_ratings.columns.tolist()
    
    return Y, Y_norm, Y_mean, book_list, user_list

# --- Simulating Pre-trained Data (X and book_list) ---
def get_pre_trained_data(num_books, num_features, book_list_from_data):
    """
    Simulates loading pre-trained book features (X) and book list.
    In a real application, X would come from a trained model's output (e.g., from the TensorFlow code),
    and book_list would be loaded alongside it to ensure consistency.

    Args:
        num_books (int): Number of books.
        num_features (int): Number of features per book.
        book_list_from_data (list): Book list obtained from data preprocessing.

    Returns:
        tuple: (X, book_list)
               X (ndarray): Matrix of book features.
               book_list (list): List of book titles.
    """
    # In a real scenario, X would be loaded from a file, e.g., np.load(X_FILE)
    # For now, we simulate it as random data
    X = np.random.rand(num_books, num_features) 
    
    # book_list would also come from the same source as X, ensuring consistency
    # For now, we use the one derived from the ratings data
    return X, book_list_from_data

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

# --- Make Predictions and Recommendations ---
def make_predictions_and_recommendations(X, W, b, Y_mean, book_list, test_ratings, test_rated_indices, num_recommendations):
    """
    Generates predicted ratings for all books and provides recommendations for the user.

    Args:
        X (ndarray): Matrix of book features.
        W (ndarray): Learned weights for the user.
        b (float): Learned bias for the user.
        Y_mean (ndarray): Mean rating for each book.
        book_list (list): List of all book titles.
        test_ratings (ndarray): Array of the user's original ratings.
        test_rated_indices (list): Indices of books the user has already rated.
        num_recommendations (int): Number of top recommendations to display.
    """
    # Calculate predictions relative to the mean
    user_predictions_norm = np.matmul(X, W) + b
    
    # Restore the mean to get actual predicted ratings
    user_predictions = user_predictions_norm + Y_mean

    # Sort predictions by rating in descending order
    # np.argsort returns indices that would sort an array; [::-1] reverses it for descending order
    sorted_indices = np.argsort(user_predictions)[::-1]

    print(f"\n--- Top {num_recommendations} Recommendations ---")
    recommended_count = 0
    for idx in sorted_indices:
        if idx not in test_rated_indices: # Only recommend books the user hasn't rated
            print(f"Predicting rating {user_predictions[idx]:0.2f} for book: {book_list[idx]}")
            recommended_count += 1
            if recommended_count >= num_recommendations:
                break
    
    print("\n--- Original vs. Predicted Ratings for Books You've Rated ---")
    for idx in test_rated_indices:
        print(f"Book: {book_list[idx]}, Original Rating: {test_ratings[idx]}, Predicted Rating: {user_predictions[idx]:0.2f}")


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load and preprocess data
    Y_original, Y_norm, Y_mean, book_list, user_list = load_and_preprocess_data(RATINGS_FILE)

    if Y_original is None: # Exit if data loading failed
        exit()

    num_books, num_users = Y_original.shape

    # 2. Simulate loading pre-trained X and use book_list
    X, book_list_final = get_pre_trained_data(num_books, NUM_FEATURES, book_list)

    # 3. Define the test user's ratings
    my_books = ["gideon the ninth, tamsyn muir", "the fifth season, n.k. jemisin", 
                "return of the thief, megan whalen turner", "iron widow, xiran jay zhao",
                "catching fire, suzanne collins"]
    my_ratings = [5, 5, 4, 3, 3]

    test_ratings, test_rated_indices = prepare_user_ratings(my_books, my_ratings, book_list_final, num_books)

    print('\n--- Your Provided Ratings ---')
    for idx in test_rated_indices:
        print(f'Rated {test_ratings[idx]} for: {book_list_final[idx]}')

    # 4. Train the linear regression model for the user
    W_user, b_user = train_linear_regression_model(X, Y_mean, test_ratings)

    # 5. Make predictions and display recommendations
    make_predictions_and_recommendations(X, W_user, b_user, Y_mean, book_list_final, test_ratings, test_rated_indices, TOP_N_RECOMMENDATIONS)
