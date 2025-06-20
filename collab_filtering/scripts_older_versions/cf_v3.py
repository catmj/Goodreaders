import numpy as np # type: ignore
import pandas as pd # type: ignore
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
import os 
import math # Import math for infinity

# --- Set Seeds for Reproducibility ---
np.random.seed(13)
tf.random.set_seed(42)

# --- Configuration ---
DATA_PATH = '../train_test_split/split_data_combined/'
RATINGS_FILE = DATA_PATH + 'train_ratings.csv'
NUM_FEATURES = 300
LEARNING_RATE = 1e-1
ITERATIONS = 400
LAMBDA = 3.25
TOP_N_RECOMMENDATIONS = 20
TRAINED_DATA_FOLDER = 'trained_data/'
RANDOM_STARTS = 1 # Number of random starts for training

# --- Early Stopping Configuration ---
VALIDATION_SPLIT_RATIO = 0.15 # Percentage of known ratings to use for validation
EARLY_STOP_PATIENCE = 5     # Number of validation checks with no improvement before stopping
VALIDATION_CHECK_INTERVAL = 20 # Check validation loss every N iterations

os.makedirs(TRAINED_DATA_FOLDER, exist_ok=True)

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(ratings_file, validation_split_ratio):
    """
    Loads and preprocesses the ratings data, returning matrices for collaborative filtering.
    Splits the *known ratings* into training and validation sets.
    """
    try:
        ratings = pd.read_csv(ratings_file)
    except FileNotFoundError:
        print(f"Error: {ratings_file} not found. Please check the path.")
        return None, None, None, None, None, None, None, None

    ratings["book"] = ratings[['Title', 'Author']].agg(', '.join, axis=1)
    ratings = ratings[["book", "User_id", "Rating"]]

    # Filter invalid ratings and convert to integer
    ratings = ratings[ratings['Rating'] != 'Invalid rating'].copy()
    ratings['Rating'] = ratings['Rating'].astype(int)

    pivot_ratings = ratings.pivot_table(index='book', columns='User_id', values='Rating')

    Y_original = pivot_ratings.to_numpy()
    
    # R_full is 1 where Y_original is not NaN, 0 otherwise (all known ratings)
    R_full = (~np.isnan(Y_original)).astype(int) 

    Y_mean = np.nanmean(Y_original, axis=1)
    Y_norm = (Y_original.T - Y_mean).T
    Y_norm = np.nan_to_num(Y_norm) # Replace NaNs with 0 in the normalized matrix for computation

    book_list = pivot_ratings.index.tolist()
    user_list = pivot_ratings.columns.tolist()

    known_rating_indices = np.argwhere(R_full == 1)
    np.random.shuffle(known_rating_indices) # Randomly shuffle these indices

    num_val_ratings = int(len(known_rating_indices) * validation_split_ratio)
    
    val_indices = known_rating_indices[:num_val_ratings]
    train_indices = known_rating_indices[num_val_ratings:]

    R_train = np.zeros_like(R_full)
    for r, c in train_indices:
        R_train[r, c] = 1

    R_val = np.zeros_like(R_full)
    for r, c in val_indices:
        R_val[r, c] = 1
    
    # Sanity check: Ensure no overlap and union is R_full
    assert np.all((R_train + R_val) == R_full), "Training and validation masks do not sum to full mask!"
    assert np.all((R_train * R_val) == 0), "Training and validation masks overlap!"

    print(f"Data split: {R_train.sum()} training ratings, {R_val.sum()} validation ratings.")

    return Y_original, R_train, R_val, Y_norm, Y_mean, book_list, user_list

# --- Collaborative Filtering Cost Function ---
@tf.function(jit_compile=True) # Added XLA compilation
def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    """
    Computes the cost for the collaborative filtering model.
    All inputs are expected to be TensorFlow tensors.
    """
    predictions = tf.linalg.matmul(X, tf.transpose(W)) + b
    error = (predictions - Y) * R
    J = 0.5 * tf.reduce_sum(tf.square(error)) + \
        (lambda_ / 2) * (tf.reduce_sum(tf.square(X)) + tf.reduce_sum(tf.square(W)))
    return J

# --- Model Training ---
def train_model(Y_norm_np, R_train_np, R_val_np, num_books, num_users, num_features, learning_rate, iterations, lambda_):
    """
    Trains the collaborative filtering model using gradient descent with early stopping.
    This function will run the training multiple times with random starts
    and return the X, W, b from the run with the lowest final validation loss.
    """
    # Convert NumPy arrays to TensorFlow tensors once at the beginning
    Y_norm_tf = tf.constant(Y_norm_np, dtype=tf.float64)
    R_train_tf = tf.constant(R_train_np, dtype=tf.float64)
    R_val_tf = tf.constant(R_val_np, dtype=tf.float64)

    best_X_overall, best_W_overall, best_b_overall = None, None, None
    min_final_val_loss_overall = math.inf # Initialize with a very high value

    print(f"\n--- Starting Model Training with {RANDOM_STARTS} Random Starts (Early Stopping Enabled) ---")

    for start_num in range(1, RANDOM_STARTS + 1):
        print(f"\n--- Training Run {start_num}/{RANDOM_STARTS} ---")

        # Re-initialize W, X, b for each random start
        W = tf.Variable(tf.random.normal((num_users, num_features), dtype=tf.float64), name='W')
        X = tf.Variable(tf.random.normal((num_books, num_features), dtype=tf.float64), name='X')
        b = tf.Variable(tf.random.normal((1, num_users), dtype=tf.float64), name='b')
        

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # --- Early Stopping variables for the current run ---
        best_val_loss_for_run = math.inf
        patience_counter = 0
        current_best_X, current_best_W, current_best_b = None, None, None # To store best for this run

        for iter_count in range(iterations):
            with tf.GradientTape() as tape:
                # Use R_train for training loss calculation
                cost_value = cofi_cost_func_v(X, W, b, Y_norm_tf, R_train_tf, lambda_)

            grads = tape.gradient(cost_value, [X, W, b])
            optimizer.apply_gradients(zip(grads, [X, W, b]))

            # --- Early Stopping Check ---
            if iter_count % VALIDATION_CHECK_INTERVAL == 0:
                # Calculate validation loss using R_val
                val_cost = cofi_cost_func_v(X, W, b, Y_norm_tf, R_val_tf, lambda_).numpy()
                
                print(f"  Iteration {iter_count}: Train Cost = {cost_value.numpy():0.4f}, Val Cost = {val_cost:0.4f}")

                if val_cost < best_val_loss_for_run:
                    best_val_loss_for_run = val_cost
                    patience_counter = 0
                    # Store current best model for this run
                    current_best_X = X.read_value()
                    current_best_W = W.read_value()
                    current_best_b = b.read_value()
                else:
                    patience_counter += 1
                    if patience_counter >= EARLY_STOP_PATIENCE:
                        print(f"  Early stopping triggered at iteration {iter_count}. Validation loss not improved for {EARLY_STOP_PATIENCE} checks.")
                        break # Break inner training loop
            else:
                # Print training cost only if not a validation check iteration
                if iter_count % 20 == 0: # Still print training cost every 20 iterations
                    print(f"  Iteration {iter_count}: Train Cost = {cost_value.numpy():0.4f}")

        # After each run, check if this run's best validation loss is the overall best
        # Ensure current_best_X, etc., are set even if no improvement was found in the initial checks.
        # This fallback handles cases where best_val_loss_for_run remains inf if VALIDATION_CHECK_INTERVAL is large.
        if current_best_X is None: # If no validation check triggered or no improvement ever happened
            current_best_X = X.read_value()
            current_best_W = W.read_value()
            current_best_b = b.read_value()
            best_val_loss_for_run = cofi_cost_func_v(X, W, b, Y_norm_tf, R_val_tf, lambda_).numpy()


        print(f"--- Training Run {start_num} Complete. Best Val Loss for run: {best_val_loss_for_run:0.4f} ---")

        if best_val_loss_for_run < min_final_val_loss_overall:
            min_final_val_loss_overall = best_val_loss_for_run
            best_X_overall = current_best_X
            best_W_overall = current_best_W
            best_b_overall = current_best_b
            print(f"  --> New overall best model found with Val loss {min_final_val_loss_overall:0.4f}")

    print(f"\n--- All Random Starts Complete. Overall Best Validation Loss: {min_final_val_loss_overall:0.4f} ---")
    
    # Return the overall best X, W, b found across all random starts as TensorFlow Tensors
    return best_X_overall, best_W_overall, best_b_overall

# --- Prediction and Recommendation ---
def make_predictions_and_recommendations(X_tf, W_tf, b_tf, Y_mean_np, Y_original_np, book_list, user_index, num_recommendations):
    """
    Makes predictions and provides recommendations for a specific user.
    X_tf, W_tf, b_tf are expected to be TensorFlow Tensors/Variables.
    Y_mean_np, Y_original_np are expected to be NumPy arrays.
    """
    # Convert Y_mean to TensorFlow tensor once
    Y_mean_tf = tf.constant(Y_mean_np, dtype=tf.float64)

    # Perform prediction entirely in TensorFlow
    p_tf = tf.linalg.matmul(X_tf, tf.transpose(W_tf)) + b_tf
    pm_tf = tf.transpose(tf.transpose(p_tf) + Y_mean_tf) # Restore the mean

    user_predictions_tf = pm_tf[:, user_index]

    # Convert user_predictions to NumPy only when needed for non-TF operations/display
    user_predictions_np = user_predictions_tf.numpy()

    # Get books the user has already rated (Y_original_np is NumPy)
    already_rated_indices = np.where(Y_original_np[:, user_index] > 0)[0]

    # Sort predictions in descending order (tf.argsort is efficient on TF tensor)
    sorted_indices = tf.argsort(user_predictions_tf, direction='DESCENDING').numpy()

    print(f"\n--- Recommendations for User {user_list[user_index]} ---")
    recommended_count = 0
    for idx in sorted_indices:
        if idx not in already_rated_indices:
            print(f"Predicting rating {user_predictions_np[idx]:0.2f} for book: {book_list[idx]}")
            recommended_count += 1
            if recommended_count >= num_recommendations:
                break
    
    print("\n--- Original vs. Predicted Ratings for User's Rated Books ---")
    for idx in already_rated_indices:
        # Only print if the original rating is available (not NaN or 0 if it was 0 for unrated)
        if Y_original_np[idx, user_index] > 0:
            print(f"Book: {book_list[idx]}, Original Rating: {Y_original_np[idx, user_index]}, Predicted Rating: {user_predictions_np[idx]:0.2f}")

# --- Data Export (Adjusted to save into a specific folder) ---
def export_data(X_tf, book_list, Y_mean_np, num_features, output_folder):
    """
    Exports the matrix X to a .npy file, book_list to a plain text file,
    and Y_mean to a .npy file, within a specified output folder.
    X_tf is expected to be a TensorFlow Tensor.
    Y_mean_np is expected to be a NumPy array.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Construct full file paths
    x_filepath = os.path.join(output_folder, f'X_features_{num_features}.npy')
    book_list_filepath = os.path.join(output_folder, f'book_list_features_{num_features}.txt')
    y_mean_filepath = os.path.join(output_folder, f'Y_mean_features_{num_features}.npy')

    # Convert X to a NumPy array for saving
    np.save(x_filepath, X_tf.numpy())

    with open(book_list_filepath, 'w', encoding='utf-8') as f:
        for book in book_list:
            f.write(book + '\n')

    np.save(y_mean_filepath, Y_mean_np) # Y_mean_np is already NumPy

    print(f"\n--- Data Export Complete ---")
    print(f"Matrix X exported to: {x_filepath}")
    print(f"Book list exported to: {book_list_filepath}")
    print(f"Y_mean exported to: {y_mean_filepath}")


# --- Main Execution ---
if __name__ == "__main__":
    # Load and preprocess data, now returning R_train and R_val
    Y_original, R_train, R_val, Y_norm, Y_mean, book_list, user_list = \
        load_and_preprocess_data(RATINGS_FILE, VALIDATION_SPLIT_RATIO)

    if Y_original is None:
        exit() # Exit if data loading failed

    num_books, num_users = Y_original.shape

    # Choose a test user (e.g., the first user, index 0)
    test_user_index = 0
    print(f"Analyzing user: {user_list[test_user_index]}")

    # Pass R_train and R_val to the training function
    X_tf, W_tf, b_tf = train_model(Y_norm, R_train, R_val, num_books, num_users, NUM_FEATURES, LEARNING_RATE, ITERATIONS, LAMBDA)

    # Pass TensorFlow Tensors to make_predictions_and_recommendations
    make_predictions_and_recommendations(X_tf, W_tf, b_tf, Y_mean, Y_original, book_list, test_user_index, TOP_N_RECOMMENDATIONS)

    # Export the matrix X (TensorFlow Tensor), book_list, and Y_mean (NumPy array) into the specified folder
    export_data(X_tf, book_list, Y_mean, NUM_FEATURES, TRAINED_DATA_FOLDER)