import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
from tqdm import tqdm # Import tqdm for progress bars
import pickle # Import pickle for saving/loading checkpoints
from datetime import datetime # Import datetime for timestamping prints

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# --- Configuration Class ---
class Config:
    """
    Configuration parameters for the collaborative filtering model training.
    """
    DATA_PATH = '../train_test_split/split_data_combined/'
    TRAIN_RATINGS_FILE = DATA_PATH + 'train_ratings.csv'
    CROSS_VAL_RATINGS_FILE = DATA_PATH + 'cross_val_ratings.csv'
    LEARNING_RATE = 0.005
    ITERATIONS = 1000
    TOP_N_RECOMMENDATIONS = 20 # Not directly used in this training script, but kept for consistency
    TRAINED_DATA_FOLDER = 'trained_data/'
    # Dynamically set checkpoint filename using os.path.join for path robustness
    TRAINING_CHECKPOINT_FILENAME = os.path.join(TRAINED_DATA_FOLDER, 'training_progress_checkpoint.pkl')

    # Final model parameters
    NUM_FEATURES = 600
    LAMBDA_VAL = 5

os.makedirs(Config.TRAINED_DATA_FOLDER, exist_ok=True)

# --- Checkpointing Helper Functions ---
def save_checkpoint(data, filename):
    """
    Saves the given data to a file using pickle for checkpointing.

    Args:
        data (dict): The data (e.g., training progress) to be saved.
        filename (str): The full path including filename where the data will be saved.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True) # Ensure directory exists
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint saved to {filename}")
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error saving checkpoint to {filename}: {e}")

def load_checkpoint(filename):
    """
    Loads checkpoint data from a file using pickle.

    Args:
        filename (str): The full path including filename from where the data will be loaded.

    Returns:
        dict: The loaded data if the file exists and can be loaded, otherwise an empty dictionary.
    """
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint loaded from {filename}")
            return data
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error loading checkpoint from {filename}: {e}. Starting fresh.")
            return {}
    return {} # Return empty dict if file doesn't exist


# --- Data Loading and Preprocessing for Combined Training Data ---
def load_and_preprocess_combined_training_data(train_file, cv_file):
    """
    Loads and preprocesses the combined training and cross-validation ratings data.

    This function reads ratings from both the training and cross-validation CSV files,
    concatenates them, and then performs the following preprocessing steps:
    1. Combines 'Title' and 'Author' to create a unique 'book' identifier.
    2. Filters out 'Invalid rating' entries and converts ratings to integers.
    3. Drops duplicate (book, user) ratings, keeping the first occurrence.
    4. Pivots the DataFrame to create a matrix where rows are books and columns are users,
       with ratings as values (NaN for unrated).
    5. Creates the observed ratings matrix R (1 for observed, 0 for unobserved).
    6. Computes the mean rating for each book (Y_mean).
    7. Normalizes the ratings by subtracting the book means (Y_norm), replacing NaNs with 0.

    Args:
        train_file (str): Path to the training ratings CSV file.
        cv_file (str): Path to the cross-validation ratings CSV file.

    Returns:
        tuple: A tuple containing:
            - Y (numpy.ndarray): Original combined ratings matrix (with NaNs).
            - R (numpy.ndarray): Binary matrix indicating observed ratings (1) or unobserved (0).
            - Y_norm (numpy.ndarray): Normalized ratings matrix (NaNs replaced with 0).
            - Y_mean (numpy.ndarray): Mean rating for each book.
            - book_list (list): List of unique book identifiers.
            - user_list (list): List of unique user IDs.
    """
    try:
        train_ratings = pd.read_csv(train_file)
        cv_ratings = pd.read_csv(cv_file)
        # Concatenate the two dataframes
        ratings = pd.concat([train_ratings, cv_ratings], ignore_index=True)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Combined training data from {train_file} and {cv_file}.")
    except FileNotFoundError as e:
        print(f"Error: One of the files ({train_file}, {cv_file}) not found. Please check the paths. Error: {e}")
        return None, None, None, None, None, None

    ratings["book"] = ratings[['Title', 'Author']].agg(', '.join, axis=1)
    ratings = ratings[["book", "User_id", "Rating"]]

    # Filter invalid ratings and convert to integer
    ratings = ratings[ratings['Rating'] != 'Invalid rating'].copy()
    ratings['Rating'] = ratings['Rating'].astype(int)

    # Drop duplicate ratings for the same (book, user) pair, keeping the first one
    # This is important after concatenation to avoid bias in pivot_table
    ratings.drop_duplicates(subset=['book', 'User_id'], inplace=True)

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
@tf.function(jit_compile=True)
def cofi_cost_func_v(X, W, b_user, b_item, Y, R, lambda_):
    """
    Computes the regularized cost for the collaborative filtering model.

    This function calculates the cost based on the predicted ratings and actual ratings,
    applying regularization to the item features (X), user features (W), and item biases (b_item).

    Args:
        X (tf.Tensor): Tensor of item features (num_books, num_features).
        W (tf.Tensor): Tensor of user features (num_users, num_features).
        b_user (tf.Tensor): Tensor of user biases (1, num_users).
        b_item (tf.Tensor): Tensor of item biases (num_books, 1).
        Y (tf.Tensor): Original ratings matrix (num_books, num_users).
        R (tf.Tensor): Binary matrix indicating observed ratings (num_books, num_users).
        lambda_ (float): Regularization parameter.

    Returns:
        tf.Tensor: The computed collaborative filtering cost (scalar).
    """
    predictions = tf.linalg.matmul(X, tf.transpose(W)) + b_user + b_item
    error = (predictions - Y) * R # Only consider observed ratings for error
    J = 0.5 * tf.reduce_sum(error**2) + \
        (lambda_ / 2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2) + \
                         tf.reduce_sum(b_item**2)) 
    return J

# --- Model Training ---
def train_model(Y_norm_train_tf, R_train_tf,
                num_books, num_users, num_features, learning_rate, iterations, lambda_):
    """
    Trains the collaborative filtering model using full-batch gradient descent.

    This function initializes the model parameters (X, W, b_user, b_item) and
    then iteratively updates them using the Adam optimizer to minimize the cost function.
    Training progress, including the loss, is printed periodically.

    Args:
        Y_norm_train_tf (tf.Tensor): Normalized training ratings matrix.
        R_train_tf (tf.Tensor): Binary matrix indicating observed ratings in the training set.
        num_books (int): Total number of unique books in the combined training data.
        num_users (int): Total number of unique users in the combined training data.
        num_features (int): The number of latent features for items and users.
        learning_rate (float): The learning rate for the Adam optimizer.
        iterations (int): The number of gradient descent iterations.
        lambda_ (float): The regularization parameter used in the cost function.

    Returns:
        tuple: A tuple containing the learned TensorFlow Variables:
            - X (tf.Variable): Learned item features.
            - W (tf.Variable): Learned user features.
            - b_user (tf.Variable): Learned user biases.
            - b_item (tf.Variable): Learned item biases.
    """
    # Set Initial Parameters (W, X, b_user, b_item)
    # Using tf.float32 for faster computation
    W = tf.Variable(tf.random.normal((num_users, num_features), dtype=tf.float32), name='W')
    X = tf.Variable(tf.random.normal((num_books, num_features), dtype=tf.float32), name='X')
    b_user = tf.Variable(tf.random.normal((1, num_users), dtype=tf.float32), name='b_user')
    b_item = tf.Variable(tf.random.normal((num_books, 1), dtype=tf.float32), name='b_item')

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    print("\n--- Starting Model Training (Full-Batch Gradient Descent) ---")
    # Wrap the training loop with tqdm for a progress bar
    for iter_count in tqdm(range(iterations), desc="Training Model", unit="iteration"):
        with tf.GradientTape() as tape:
            # Calculate training cost
            training_cost = cofi_cost_func_v(X, W, b_user, b_item,
                                             Y_norm_train_tf,
                                             R_train_tf,
                                             lambda_)

        grads = tape.gradient(training_cost, [X, W, b_user, b_item])
        optimizer.apply_gradients(zip(grads, [X, W, b_user, b_item]))

        # Print loss updates less frequently to keep the progress bar clean
        if iter_count % max(1, iterations // 10) == 0: # Print 10 times during training
            tqdm.write(f"Iter {iter_count}: Train Loss: {training_cost:0.4f}")
    print("--- Model Training Complete ---")

    return X, W, b_user, b_item

# --- Data Export (Updated to include lambda_val in filenames) ---
def export_data(X, b_item, book_list, Y_mean, num_features, output_folder, lambda_val):
    """
    Exports the learned model parameters and related data to files.

    This function saves the learned item features (X), item biases (b_item),
    the list of book identifiers, and the book mean ratings to specified files.

    Args:
        X (tf.Tensor): Learned item features.
        b_item (tf.Tensor): Learned item biases.
        book_list (list): List of unique book identifiers.
        Y_mean (numpy.ndarray): Mean rating for each book.
        num_features (int): The number of latent features used in the model.
        output_folder (str): The directory where the files will be saved.
        lambda_val (float): The regularization parameter used, included in filenames for identification.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Construct full file paths, including lambda_val for uniqueness
    x_filepath = os.path.join(output_folder, f'X_features_{num_features}_lambda_{lambda_val}.npy')
    b_item_filepath = os.path.join(output_folder, f'b_item_bias_{num_features}_lambda_{lambda_val}.npy')
    book_list_filepath = os.path.join(output_folder, f'book_list_features_{num_features}_lambda_{lambda_val}.txt')
    y_mean_filepath = os.path.join(output_folder, f'Y_mean_features_{num_features}_lambda_{lambda_val}.npy')

    # Convert tensors to NumPy arrays
    np.save(x_filepath, X.numpy())
    np.save(b_item_filepath, b_item.numpy())

    with open(book_list_filepath, 'w', encoding='utf-8') as f:
        for book in book_list:
            f.write(book + '\n')

    np.save(y_mean_filepath, Y_mean)

    print(f"\n--- Data Export Complete for Features={num_features}, Lambda={lambda_val} ---")
    print(f"Required model parameters (X, b_item, book_list, Y_mean) saved to: {output_folder}")

# --- Main Execution ---
if __name__ == "__main__":
    # --- Load Combined Training Data (Train + Cross-Validation) ---
    Y_combined_original_np, R_combined_np, Y_combined_norm_np, Y_combined_mean_np, book_list_combined, user_list_combined = \
        load_and_preprocess_combined_training_data(Config.TRAIN_RATINGS_FILE, Config.CROSS_VAL_RATINGS_FILE)

    if Y_combined_original_np is None:
        print("Combined training data loading failed. Exiting.")
        exit() # Exit if data loading failed

    num_books_combined, num_users_combined = Y_combined_original_np.shape

    Y_combined_norm_tf = tf.constant(Y_combined_norm_np, dtype=tf.float32)
    R_combined_tf = tf.constant(R_combined_np, dtype=tf.float32)

    print(f"\n######################################################")
    print(f"### Running Final Model Training: Features={Config.NUM_FEATURES}, Lambda={Config.LAMBDA_VAL} ###")
    print(f"######################################################")

    # Train the model with the specified parameters using the combined data
    X, W, b_user, b_item = train_model(
        Y_combined_norm_tf, R_combined_tf,    # Combined training and cross-validation data
        num_books_combined, num_users_combined, # Dimensions based on combined set
        Config.NUM_FEATURES, Config.LEARNING_RATE, Config.ITERATIONS, Config.LAMBDA_VAL
    )

    # Export the learned parameters for this final configuration
    export_data(X, b_item, book_list_combined, Y_combined_mean_np,
                Config.NUM_FEATURES, Config.TRAINED_DATA_FOLDER, Config.LAMBDA_VAL)

    print(f"\n--- Final Model Training (Features={Config.NUM_FEATURES}, Lambda={Config.LAMBDA_VAL}) Complete ---\n")

    print("\nFinal model generation complete.")
