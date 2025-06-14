import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
from tqdm import tqdm # Import tqdm for progress bars
import pickle # Import pickle for saving/loading checkpoints
from datetime import datetime # Import datetime for timestamping prints

# --- Configuration (These global variables will be overridden by loop parameters) ---
DATA_PATH = '../train_test_split/split_data_combined/'
RATINGS_FILE = DATA_PATH + 'train_ratings.csv'
LEARNING_RATE = 0.005 # Stays global or can be added to config
ITERATIONS = 1000 # Number of gradient descent iterations
TOP_N_RECOMMENDATIONS = 20
TRAINED_DATA_FOLDER = 'trained_data/'
TRAINING_CHECKPOINT_FILENAME = os.path.join(TRAINED_DATA_FOLDER, 'training_progress_checkpoint.pkl')


os.makedirs(TRAINED_DATA_FOLDER, exist_ok=True)

# --- Checkpointing Helper Functions ---
def save_checkpoint(data, filename):
    """Saves the given data to a file using pickle."""
    os.makedirs(os.path.dirname(filename), exist_ok=True) # Ensure directory exists
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint saved to {filename}")
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error saving checkpoint to {filename}: {e}")

def load_checkpoint(filename):
    """Loads checkpoint data from a file using pickle."""
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
@tf.function(jit_compile=True)
def cofi_cost_func_v(X, W, b_user, b_item, Y, R, lambda_):
    """
    Computes the cost for the collaborative filtering model, now including user and item biases.
    Operates on full matrices (no mini-batching here).
    """
    predictions = tf.linalg.matmul(X, tf.transpose(W)) + b_user + b_item
    error = (predictions - Y) * R
    J = 0.5 * tf.reduce_sum(error**2) + \
        (lambda_ / 2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2) + tf.reduce_sum(b_item**2))
    return J

# --- Model Training ---
def train_model(Y_norm_tf, R_tf, num_books, num_users, num_features, learning_rate, iterations, lambda_):
    """
    Trains the collaborative filtering model using full-batch gradient descent.
    Accepts Y_norm_tf and R_tf directly as TensorFlow tensors.
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
            # Ensure tensors passed to cost function have matching dtype
            cost_value = cofi_cost_func_v(X, W, b_user, b_item, 
                                          Y_norm_tf, # Pass full Y_norm_tf
                                          R_tf,      # Pass full R_tf
                                          lambda_)

        grads = tape.gradient(cost_value, [X, W, b_user, b_item])
        optimizer.apply_gradients(zip(grads, [X, W, b_user, b_item]))

        # Print loss updates less frequently to keep the progress bar clean
        if iter_count % max(1, iterations // 10) == 0: # Print 10 times during training
            tqdm.write(f"Training loss at iteration {iter_count}: {cost_value:0.4f}")
    print("--- Model Training Complete ---")

    return X, W, b_user, b_item

# --- Data Export (Updated to include lambda_val in filenames) ---
def export_data(X, b_item, book_list, Y_mean, num_features, output_folder, lambda_val):
    """
    Exports the learned model parameters (X, b_item), book_list,
    and Y_mean to files within a specified output folder.
    W and b_user are NOT exported.
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

    np.save(y_mean_filepath, Y_mean) # Y_mean is already NumPy float64 from preprocessing, export as is.

    print(f"\n--- Data Export Complete for Features={num_features}, Lambda={lambda_val} ---")
    print(f"Required model parameters (X, b_item, book_list, Y_mean) saved to: {output_folder}")

# --- Main Execution ---
if __name__ == "__main__":
    # Define your experiment configurations
    experiment_configs = [

        # New configs to combat underfitting:
        # Higher number of features (increased model capacity)
        {'num_features': 500, 'lambda_val': 15},
        {'num_features': 600, 'lambda_val': 15},
        {'num_features': 800, 'lambda_val': 15}, # Even higher features

        # Lower lambda values (reduced regularization)
        {'num_features': 400, 'lambda_val': 10}, # Less regularization
        {'num_features': 400, 'lambda_val': 5},  # Even less regularization
        {'num_features': 400, 'lambda_val': 1},  # Minimal regularization

        # Combination of higher features and lower lambda
        {'num_features': 500, 'lambda_val': 10},
        {'num_features': 600, 'lambda_val': 5},
        {'num_features': 800, 'lambda_val': 10},

        # New configurations to train:
        {'num_features': 400, 'lambda_val': 8},
        {'num_features': 500, 'lambda_val': 8},
        {'num_features': 600, 'lambda_val': 8},
        {'num_features': 600, 'lambda_val': 10}
    ]

    # Load data only once outside the loop
    Y_original_np, R_np, Y_norm_np, Y_mean_np, book_list, user_list = load_and_preprocess_data(RATINGS_FILE)

    if Y_original_np is None:
        exit() # Exit if data loading failed

    num_books, num_users = Y_original_np.shape

    # Convert NumPy arrays to TensorFlow tensors ONCE after loading
    # Cast to tf.float32 for consistency with model training
    Y_norm_tf_global = tf.constant(Y_norm_np, dtype=tf.float32)
    R_tf_global = tf.constant(R_np, dtype=tf.float32)


    # Load training progress checkpoint
    training_progress = load_checkpoint(TRAINING_CHECKPOINT_FILENAME)
    if training_progress:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Resuming training from checkpoint. Already completed {len(training_progress)} configurations.")

    # Loop through each experiment configuration
    for config in experiment_configs:
        current_num_features = config['num_features']
        current_lambda = config['lambda_val']
        config_key = (current_num_features, current_lambda) # Use tuple as a unique key for checkpoint

        if training_progress.get(config_key, False):
            print(f"\n--- Skipping Experiment: Features={current_num_features}, Lambda={current_lambda} (already completed) ---")
            continue

        print(f"\n######################################################")
        print(f"### Running Experiment: Features={current_num_features}, Lambda={current_lambda} ###")
        print(f"######################################################")

        # Train the model with the current configuration
        # Pass the pre-converted TensorFlow tensors directly
        X, W, b_user, b_item = train_model(
            Y_norm_tf_global, R_tf_global, # Pass global TF tensors for full-batch training
            num_books, num_users,
            current_num_features, LEARNING_RATE, ITERATIONS, current_lambda
        )

        # Export the learned parameters for the current configuration
        # Note: Y_mean_np is a NumPy array, no change needed.
        export_data(X, b_item, book_list, Y_mean_np, current_num_features, TRAINED_DATA_FOLDER, current_lambda)

        print(f"\n--- Experiment Features={current_num_features}, Lambda={current_lambda} Complete ---\n")

        # Save progress after successful completion of an experiment
        training_progress[config_key] = True
        save_checkpoint(training_progress, TRAINING_CHECKPOINT_FILENAME)

    print("\nAll experiments complete.")
