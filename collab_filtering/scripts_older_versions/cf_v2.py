import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os 

# --- Configuration ---
DATA_PATH = '../train_test_split/split_data_combined/'
RATINGS_FILE = DATA_PATH + 'train_ratings.csv'
NUM_FEATURES = 300
LEARNING_RATE = 1e-1
ITERATIONS = 400
LAMBDA = 3.25
TOP_N_RECOMMENDATIONS = 20
TRAINED_DATA_FOLDER = 'trained_data/'

os.makedirs(TRAINED_DATA_FOLDER, exist_ok=True)

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(ratings_file):
    """
    Loads and preprocesses the ratings data, returning matrices for collaborative filtering.
    """
    try:
        ratings = pd.read_csv(ratings_file)
    except FileNotFoundError:
        print(f"Error: {ratings_file} not found. Please check the path.")
        return None, None, None, None, None, None

    ratings["book"] = ratings[['Title', 'Author']].agg(', '.join, axis=1)
    ratings = ratings[["book", "User_id", "Rating"]]

    # Filter invalid ratings and convert to integer
    ratings = ratings[ratings['Rating'] != 'Invalid rating'].copy()
    ratings['Rating'] = ratings['Rating'].astype(int)

    pivot_ratings = ratings.pivot_table(index='book', columns='User_id', values='Rating')

    Y = pivot_ratings.to_numpy()
    R = (~np.isnan(Y)).astype(int) # R is 1 where Y is not NaN, 0 otherwise

    Y_mean = np.nanmean(Y, axis=1)
    Y_norm = (Y.T - Y_mean).T
    Y_norm = np.nan_to_num(Y_norm) # Replace NaNs with 0 in the normalized matrix for computation

    book_list = pivot_ratings.index.tolist()
    user_list = pivot_ratings.columns.tolist()

    return Y, R, Y_norm, Y_mean, book_list, user_list

# --- Collaborative Filtering Cost Function ---
@tf.function(jit_compile=True) # Added XLA compilation
def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    """
    Computes the cost for the collaborative filtering model.
    """
    predictions = tf.linalg.matmul(X, tf.transpose(W)) + b
    error = (predictions - Y) * R
    J = 0.5 * tf.reduce_sum(error**2) + \
        (lambda_ / 2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J

# --- Model Training ---
def train_model(Y_norm, R, num_books, num_users, num_features, learning_rate, iterations, lambda_):
    """
    Trains the collaborative filtering model using gradient descent.
    """
    # Set Initial Parameters (W, X, b), use tf.Variable to track these variables
    W = tf.Variable(tf.random.normal((num_users, num_features), dtype=tf.float64), name='W')
    X = tf.Variable(tf.random.normal((num_books, num_features), dtype=tf.float64), name='X')
    b = tf.Variable(tf.random.normal((1, num_users), dtype=tf.float64), name='b')

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    print("\n--- Starting Model Training ---")
    for iter_count in range(iterations):
        with tf.GradientTape() as tape:
            cost_value = cofi_cost_func_v(X, W, b, tf.constant(Y_norm, dtype=tf.float64), tf.constant(R, dtype=tf.float64), lambda_)

        grads = tape.gradient(cost_value, [X, W, b])
        optimizer.apply_gradients(zip(grads, [X, W, b]))

        if iter_count % 20 == 0:
            print(f"Training loss at iteration {iter_count}: {cost_value:0.4f}")
    print("--- Model Training Complete ---")
    return X, W, b

# --- Prediction and Recommendation ---
def make_predictions_and_recommendations(X, W, b, Y_mean, Y_original, book_list, user_index, num_recommendations):
    """
    Makes predictions and provides recommendations for a specific user.
    """
    p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()
    pm = (p.T + Y_mean).T # Restore the mean

    user_predictions = pm[:, user_index]

    # Get books the user has already rated
    already_rated_indices = np.where(Y_original[:, user_index] > 0)[0]

    # Sort predictions in descending order
    sorted_indices = tf.argsort(user_predictions, direction='DESCENDING').numpy()

    print(f"\n--- Recommendations for User {user_list[user_index]} ---")
    recommended_count = 0
    for idx in sorted_indices:
        if idx not in already_rated_indices:
            print(f"Predicting rating {user_predictions[idx]:0.2f} for book: {book_list[idx]}")
            recommended_count += 1
            if recommended_count >= num_recommendations:
                break
    
    print("\n--- Original vs. Predicted Ratings for User's Rated Books ---")
    for idx in already_rated_indices:
        # Only print if the original rating is available (not NaN or 0 if it was 0 for unrated)
        if Y_original[idx, user_index] > 0:
            print(f"Book: {book_list[idx]}, Original Rating: {Y_original[idx, user_index]}, Predicted Rating: {user_predictions[idx]:0.2f}")

# --- Data Export (Adjusted to save into a specific folder) ---
def export_data(X, book_list, Y_mean, num_features, output_folder):
    """
    Exports the matrix X to a .npy file, book_list to a plain text file,
    and Y_mean to a .npy file, within a specified output folder.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Construct full file paths
    x_filepath = os.path.join(output_folder, f'X_features_{num_features}.npy')
    book_list_filepath = os.path.join(output_folder, f'book_list_features_{num_features}.txt')
    y_mean_filepath = os.path.join(output_folder, f'Y_mean_features_{num_features}.npy')

    np.save(x_filepath, X.numpy())

    with open(book_list_filepath, 'w', encoding='utf-8') as f:
        for book in book_list:
            f.write(book + '\n')

    np.save(y_mean_filepath, Y_mean)

    print(f"\n--- Data Export Complete ---")
    print(f"Matrix X exported to: {x_filepath}")
    print(f"Book list exported to: {book_list_filepath}")
    print(f"Y_mean exported to: {y_mean_filepath}")


# --- Main Execution ---
if __name__ == "__main__":
    Y_original, R, Y_norm, Y_mean, book_list, user_list = load_and_preprocess_data(RATINGS_FILE)

    if Y_original is None:
        exit() # Exit if data loading failed

    num_books, num_users = Y_original.shape

    # Choose a test user (e.g., the first user, index 0)
    test_user_index = 0
    print(f"Analyzing user: {user_list[test_user_index]}")

    X, W, b = train_model(Y_norm, R, num_books, num_users, NUM_FEATURES, LEARNING_RATE, ITERATIONS, LAMBDA)

    make_predictions_and_recommendations(X, W, b, Y_mean, Y_original, book_list, test_user_index, TOP_N_RECOMMENDATIONS)

    # Export the matrix X, book_list, and Y_mean into the specified folder
    export_data(X, book_list, Y_mean, NUM_FEATURES, TRAINED_DATA_FOLDER)