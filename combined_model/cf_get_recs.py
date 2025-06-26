import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import os

# --- Configuration Constants ---
PRETRAINED_DATA_PATH = '../collab_filtering/trained_data/'
NUM_FEATURES = 600
NEW_USER_REGULARIZATION_STRENGTH = 5


# --- Loading Pre-trained Data (book_features, item_bias, all_book_identifiers, and mean_book_ratings) ---
def load_pretrained_data(
    num_features: int = NUM_FEATURES,
    regularization_strength: int = NEW_USER_REGULARIZATION_STRENGTH,
    pretrained_data_dir: str = PRETRAINED_DATA_PATH
) -> tuple[np.ndarray | None, np.ndarray | None, list | None, np.ndarray | None, int | None]:
    """
    Loads pre-trained book features (book_features), item bias (item_bias),
    the master list of book identifiers (all_book_identifiers), and mean ratings (mean_book_ratings)
    from specified files.

    Args:
        num_features (int): Number of features per book (used in filename).
        regularization_strength (int): Regularization lambda value used during pre-training (used in filename).
        pretrained_data_dir (str): The directory where the .npy and .txt files are saved.

    Returns:
        tuple: (book_features, item_bias, all_book_identifiers, mean_book_ratings, total_books)
               book_features (ndarray): Matrix of book features.
               item_bias (ndarray): Item bias vector.
               all_book_identifiers (list): List of all book identifiers.
               mean_book_ratings (ndarray): Mean rating for each book.
               total_books (int): Total number of books loaded.
               Returns (None, None, None, None, None) if files not found or an error occurs.
    """

    # Construct file paths for pre-trained data
    book_features_filename = os.path.join(
        pretrained_data_dir,
        f'X_features_{num_features}_lambda_{regularization_strength}.npy'
    )
    item_bias_filename = os.path.join(
        pretrained_data_dir,
        f'b_item_bias_{num_features}_lambda_{regularization_strength}.npy'
    )
    book_list_filename = os.path.join(
        pretrained_data_dir,
        f'book_list_features_{num_features}_lambda_{regularization_strength}.txt'
    )
    mean_ratings_filename = os.path.join(
        pretrained_data_dir,
        f'Y_mean_features_{num_features}_lambda_{regularization_strength}.npy'
    )

    book_features, item_bias, all_book_identifiers, mean_book_ratings, total_books = \
        None, None, None, None, None

    # Load book features and item bias
    try:
        book_features = np.load(book_features_filename)
        item_bias = np.load(item_bias_filename)
        total_books = book_features.shape[0]
        print(f"Loaded model parameters (book_features, item_bias) from: {pretrained_data_dir}")
    except FileNotFoundError as e:
        print(f"Error: Model parameter file not found: {e}. Please ensure X and b_item files exist in '{pretrained_data_dir}'.")
        return None, None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred while loading model parameters: {e}")
        return None, None, None, None, None

    # Load master book list (now called identifiers)
    try:
        with open(book_list_filename, 'r', encoding='utf-8') as f:
            all_book_identifiers = [line.strip() for line in f]
        print(f"Loaded book identifiers from: {book_list_filename}")
    except FileNotFoundError:
        print(f"Error: Book identifier file '{book_list_filename}' not found. Please ensure it exists.")
        return None, None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred while loading the book identifiers: {e}")
        return None, None, None, None, None

    # Load mean ratings
    try:
        mean_book_ratings = np.load(mean_ratings_filename)
        print(f"Loaded mean_book_ratings from: {mean_ratings_filename}")
    except FileNotFoundError:
        print(f"Error: mean_book_ratings file '{mean_ratings_filename}' not found. Please ensure it exists.")
        return None, None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred while loading mean ratings: {e}")
        return None, None, None, None, None

    return book_features, item_bias, all_book_identifiers, mean_book_ratings, total_books

# --- Prepare User Ratings ---
def prepare_user_ratings(
    user_rated_book_identifiers: list[str], 
    user_book_ratings: list[float],
    all_book_identifiers: list[str],        
    total_books: int
) -> tuple[np.ndarray, list[int]]:
    """
    Prepares a test user's ratings array based on their provided books and ratings.

    Args:
        user_rated_book_identifiers (list): List of book identifiers rated by the user.
        user_book_ratings (list): List of ratings corresponding to user_rated_book_identifiers.
        all_book_identifiers (list): The master list of all book identifiers.
        total_books (int): Total number of books in the master list.

    Returns:
        tuple: (user_ratings_array, user_rated_indices)
               user_ratings_array (ndarray): An array of the user's ratings, with NaN for unrated books.
               user_rated_indices (list): Indices of books the user has rated within the master list.
    """
    user_ratings_array = np.full(total_books, np.nan)
    user_rated_indices = []

    for book_identifier, rating in zip(user_rated_book_identifiers, user_book_ratings):
        try:
            # Find the index of the book in the master list
            idx = all_book_identifiers.index(book_identifier)
            user_ratings_array[idx] = rating
            user_rated_indices.append(idx)
        except ValueError:
            print(f"Warning: Book '{book_identifier}' not found in the master book list. Skipping.")
            continue

    return user_ratings_array, user_rated_indices

# --- Train Linear Regression Model (now with Ridge regularization) ---
def train_user_preference_model(
    book_features: np.ndarray,
    item_bias: np.ndarray,
    mean_book_ratings: np.ndarray,
    user_ratings_array: np.ndarray,
    regularization_strength: float
) -> tuple[np.ndarray | None, float | None]:
    """
    Trains a Ridge Regression model for a specific user to learn their preferences
    based on their ratings and book features, with L2 regularization.

    Args:
        book_features (ndarray): Matrix of all book features.
        item_bias (ndarray): Item bias vector (used in adjusting ratings).
        mean_book_ratings (ndarray): Mean rating for each book.
        user_ratings_array (ndarray): Array of the user's ratings (with NaN for unrated).
        regularization_strength (float): Regularization strength (alpha in Ridge).

    Returns:
        tuple: (user_weights, user_bias)
               user_weights (ndarray): Learned weights (coefficients) for the user's features.
               user_bias (float): Learned bias (intercept) for the user.
               Returns (None, None) if training fails due to insufficient data.
    """

    target_for_regression = user_ratings_array - mean_book_ratings - item_bias.flatten()

    # Create a mask for rated books (where target_for_regression is not NaN)
    rated_mask = ~np.isnan(target_for_regression)

    # Select features and target for only the books the user has rated
    # Ensure book_features has matching dimensions to target_for_regression (total_books x num_features)
    features_for_rated_books = book_features[rated_mask]
    target_for_rated_books = target_for_regression[rated_mask]

    # Check if there's enough valid data to train the model
    if features_for_rated_books.shape[0] == 0:
        print("Error: No valid rated books found for training the user model.")
        return None, None
    if features_for_rated_books.shape[1] == 0:
        print("Error: `features_for_rated_books` has no features. Check NUM_FEATURES configuration.")
        return None, None

    # Train the Ridge regression model
    # alpha is the regularization strength (lambda)
    ridge_model = Ridge(alpha=regularization_strength).fit(features_for_rated_books, target_for_rated_books)

    user_weights = ridge_model.coef_  
    user_bias = ridge_model.intercept_ 

    return user_weights, user_bias

# --- Function to generate user predictions ---
def generate_user_predictions(
    book_features: np.ndarray,
    item_bias: np.ndarray,
    mean_book_ratings: np.ndarray,
    all_book_identifiers: list[str],  
    total_books: int,
    user_rated_book_identifiers: list[str], 
    user_book_ratings: list[float],
    new_user_regularization_strength: float
) -> pd.DataFrame | None: 
    """
    Prepares user ratings, trains the user's linear regression model,
    and calculates predicted ratings for all books, including user-rated books
    with their original ratings and a flag indicating if they were rated.

    Args:
        book_features (ndarray): The full matrix of all book features.
        item_bias (ndarray): The full item bias vector.
        mean_book_ratings (ndarray): The mean rating for all books.
        all_book_identifiers (list): The master list of all book identifiers.
        total_books (int): Total number of books.
        user_rated_book_identifiers (list): List of book identifiers rated by the user.
        user_book_ratings (list): List of ratings corresponding to user_rated_book_identifiers.
        new_user_regularization_strength (float): Regularization parameter for the new user model.

    Returns:
        pd.DataFrame: Pandas DataFrame with 'Book Identifier', 'Predicted Rating',
                      'Rated by User', and 'Original Rating' columns.
                      Returns None if user model training fails.
    """
    # Step 1: Prepare the user's ratings array
    user_ratings_array, user_rated_indices = prepare_user_ratings(
        user_rated_book_identifiers, user_book_ratings, all_book_identifiers, total_books
    )

    print('\n--- Your Provided Ratings (against full book list) ---')
    if user_rated_indices:
        for idx in user_rated_indices:
            print(f'Rated {user_ratings_array[idx]} for: {all_book_identifiers[idx]}')
    else:
        print("No provided books were found in the main book list for training.")

    # Step 2: Train the user-specific linear regression model
    user_weights, user_bias = train_user_preference_model(
        book_features, item_bias, mean_book_ratings, user_ratings_array, new_user_regularization_strength
    )

    # Handle cases where the user model could not be trained
    if user_weights is None or user_bias is None:
        print("User-specific model could not be trained. Cannot generate full predictions.")
        return None

    # Step 3: Calculate predictions for all books
    # The `normalized_predictions` represent the user's raw preference based on features and their personal bias
    normalized_predictions = np.matmul(book_features, user_weights) + user_bias

    # Add back the global mean rating and item-specific bias to get the final predicted rating
    predicted_ratings = normalized_predictions + mean_book_ratings + item_bias.flatten()

    # Step 4: Create a Pandas DataFrame with all necessary columns
    predictions_data = []
    user_rated_books_dict = {user_rated_book_identifiers[i]: user_book_ratings[i] 
                             for i in range(len(user_rated_book_identifiers))}

    for i, book_id in enumerate(all_book_identifiers):
        is_rated = book_id in user_rated_books_dict
        original_rating = user_rated_books_dict.get(book_id, np.nan) # np.nan if not rated

        predictions_data.append({
            'Book Identifier': book_id,
            'Predicted Rating': predicted_ratings[i],
            'Rated by User': is_rated,
            'Original Rating': original_rating
        })

    predicted_ratings_df = pd.DataFrame(predictions_data)
    
    predicted_ratings_df = predicted_ratings_df.sort_values(
        by='Predicted Rating', ascending=False
    ).reset_index(drop=True)

    return predicted_ratings_df


