import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dot, Concatenate, Dense, Flatten
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tensorflow.keras.callbacks import EarlyStopping

# --- Helper function for processing user ratings file ---
def _process_user_ratings_file(file_path: str) -> pd.DataFrame:
    """
    Helper function to parse user ratings from a CSV file, combine title and author,
    and organize data into a DataFrame where each row is a unique user-book-rating interaction.

    Args:
        file_path (str): The path to the CSV file containing user ratings.

    Returns:
        pd.DataFrame: A DataFrame where each row represents a unique user-book-rating interaction and contains:
                      - 'user_id': The ID of the user.
                      - 'book_id': The combined title and author of the book.
                      - 'rating': The numerical rating given by the user to the book.
                      Returns an empty DataFrame if file reading or processing fails.
              Example:
                     user_id                   book_id            rating
              0      1494253      The Woman in the Window, A.J. Finn      3.0
              1      126589   Shopaholic Takes Manhattan, Sophie Kinsella 2.0
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {file_path}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file at {file_path} is empty.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while reading the CSV file {file_path}: {e}")
        return pd.DataFrame()

    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()

    # Ensure 'Title', 'Author', 'Rating', 'User_id' columns exist
    required_cols = ['Title', 'Author', 'Rating', 'User_id']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing one or more required columns ({required_cols}) in the CSV file.")
        return pd.DataFrame()

    # Combines Title and Author into a string of the form "{Title}, {Author}"
    # and renames 'User_id' to 'user_id' for consistency with model expectations.
    df['book_id'] = df.apply(
        lambda row: f"{row['Title'].strip()}, {row['Author'].strip()}"
        if pd.notna(row['Title']) and pd.notna(row['Author'])
        else np.nan, axis=1
    )
    df.rename(columns={'User_id': 'user_id'}, inplace=True)

    # Clean 'Rating' column: replace 'Invalid rating' with NaN and convert to numeric
    df['rating'] = pd.to_numeric(
        df['Rating'].replace('Invalid rating', np.nan), errors='coerce'
    )

    # Drop rows where 'book_id' or 'rating' is NaN (e.g., from invalid entries)
    # Also drop original 'Title', 'Author', 'Rating' columns as they are no longer needed
    df.dropna(subset=['book_id', 'rating'], inplace=True)
    df = df[['user_id', 'book_id', 'rating']] # Keep only the required columns

    # Convert User_id to string for consistency
    df['user_id'] = df['user_id'].astype(str)

    return df

# --- 1. Data Preparation ---
# Define the path to your training CSV file
USER_RATINGS_TRAIN_FILE_PATH = '../train_test_split/split_data_combined/train_ratings.csv' # <--- IMPORTANT: Update this path

# Process the raw user ratings file for the training set
df_full_train = _process_user_ratings_file(USER_RATINGS_TRAIN_FILE_PATH)

# Check if the training dataframe is empty and exit if so (no dummy data generation)
if df_full_train.empty:
    print(f"Error: No valid training data loaded from '{USER_RATINGS_TRAIN_FILE_PATH}'. Exiting.")
    sys.exit(1) # Exit with a non-zero status code to indicate an error

# --- Ensure Disjoint Users for Train/Validation Split ---
# First, get all unique user IDs from the full training data
unique_user_ids = df_full_train['user_id'].unique()

# Split these unique user IDs into training and validation user IDs
# This ensures that no user appears in both the training and validation sets.
train_user_ids, val_user_ids = train_test_split(unique_user_ids, test_size=0.2, random_state=42)

# Create the actual training and validation DataFrames based on the user ID splits
df_train_actual = df_full_train[df_full_train['user_id'].isin(train_user_ids)].copy()
df_val = df_full_train[df_full_train['user_id'].isin(val_user_ids)].copy()

if df_train_actual.empty or df_val.empty:
    print("Error: After splitting by user, either training or validation set is empty. Exiting.")
    sys.exit(1)


# Combine unique users and books from both the actual training and validation sets for consistent mapping
all_users = pd.concat([df_train_actual['user_id'], df_val['user_id']]).astype('category')
all_books = pd.concat([df_train_actual['book_id'], df_val['book_id']]).astype('category')

# Create mappings for all unique users and books
all_users_categories = all_users.cat.categories
all_books_categories = all_books.cat.categories

user_id_to_idx = {original_id: idx for idx, original_id in enumerate(all_users_categories)}
book_id_to_idx = {original_id: idx for idx, original_id in enumerate(all_books_categories)}
idx_to_user_id = {idx: original_id for original_id, idx in user_id_to_idx.items()}
idx_to_book_id = {idx: original_id for original_id, idx in book_id_to_idx.items()}

# Apply the common mappings to both actual training and validation dataframes
df_train_actual['user_idx'] = df_train_actual['user_id'].map(user_id_to_idx)
df_train_actual['book_idx'] = df_train_actual['book_id'].map(book_id_to_idx)
df_val['user_idx'] = df_val['user_id'].map(user_id_to_idx)
df_val['book_idx'] = df_val['book_id'].map(book_id_to_idx)

# Drop any rows where mapping might have failed (shouldn't happen if all IDs are in combined categories)
df_train_actual.dropna(subset=['user_idx', 'book_idx'], inplace=True)
df_val.dropna(subset=['user_idx', 'book_idx'], inplace=True)


# Extract the mapped IDs and ratings for training and cross-validation
user_input_train = df_train_actual['user_idx'].values.astype(int)
book_input_train = df_train_actual['book_idx'].values.astype(int)
y_train = df_train_actual['rating'].values

user_input_val = df_val['user_idx'].values.astype(int)
book_input_val = df_val['book_idx'].values.astype(int)
y_val = df_val['rating'].values

n_users = len(all_users_categories) # Total number of unique users across both sets
n_books = len(all_books_categories) # Total number of unique books across both sets

embedding_dim = 200 # Hyperparameter: Dimension of the embedding vectors.
                   # Common values are 32, 50, 64, or 100. Adjust based on dataset size and performance.

print(f"\nDataset Statistics:")
print(f"Number of unique users (total): {n_users}")
print(f"Number of unique books (total): {n_books}")
print(f"Total training ratings (actual training set): {len(y_train)}")
print(f"Total cross-validation ratings (disjoint users): {len(y_val)}")
print(f"Embedding Dimension: {embedding_dim}")


# --- 2. Model Architecture (NeuMF - Neural Matrix Factorization) ---
# NeuMF combines a Generalized Matrix Factorization (GMF) component
# and a Multi-Layer Perceptron (MLP) component.

def create_neumf_model(n_users, n_books, embedding_dim):
    """
    Creates the NeuMF model architecture.
    Separated into a function to allow re-initialization with potentially new n_users/n_books.
    """
    # GMF Branch: Handles element-wise product of user and item embeddings
    user_input_gmf = Input(shape=(1,), name='user_input_gmf')
    book_input_gmf = Input(shape=(1,), name='book_input_gmf')

    user_embedding_gmf = Embedding(input_dim=n_users, output_dim=embedding_dim, name='user_embedding_gmf')(user_input_gmf)
    book_embedding_gmf = Embedding(input_dim=n_books, output_dim=embedding_dim, name='book_embedding_gmf')(book_input_gmf)

    user_vec_gmf = Flatten(name='user_vec_gmf_flatten')(user_embedding_gmf)
    book_vec_gmf = Flatten(name='book_vec_gmf_flatten')(book_embedding_gmf)

    gmf_output = Dot(axes=1, name='gmf_element_wise_product')([user_vec_gmf, book_vec_gmf])

    # MLP Branch: Handles concatenation and dense neural network layers
    user_input_mlp = Input(shape=(1,), name='user_input_mlp')
    book_input_mlp = Input(shape=(1,), name='book_input_mlp')

    user_embedding_mlp = Embedding(input_dim=n_users, output_dim=embedding_dim, name='user_embedding_mlp')(user_input_mlp)
    book_embedding_mlp = Embedding(input_dim=n_books, output_dim=embedding_dim, name='book_embedding_mlp')(book_input_mlp)

    user_vec_mlp = Flatten(name='user_vec_mlp_flatten')(user_embedding_mlp)
    book_vec_mlp = Flatten(name='book_vec_mlp_flatten')(book_embedding_mlp)

    concat_mlp = Concatenate(name='mlp_concat_embeddings')([user_vec_mlp, book_vec_mlp])

    mlp_output = Dense(128, activation='relu', name='mlp_dense_1')(concat_mlp)
    mlp_output = Dense(64, activation='relu', name='mlp_dense_2')(mlp_output)
    mlp_output = Dense(32, activation='relu', name='mlp_dense_3')(mlp_output)

    # Combine GMF and MLP outputs
    combined_output = Concatenate(name='combined_gmf_mlp_output')([gmf_output, mlp_output])

    # Final prediction layer
    prediction = Dense(1, activation='linear', name='final_prediction_layer')(combined_output)

    model = Model(inputs=[user_input_gmf, book_input_gmf, user_input_mlp, book_input_mlp],
                  outputs=prediction,
                  name='NeuMF_Recommendation_Model')
    return model

model = create_neumf_model(n_users, n_books, embedding_dim)

print("\nNeuMF Model Summary (Initial Training):")
model.summary()

# --- 3. Training the Model ---
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Define EarlyStopping callback
early_stopping_callback = EarlyStopping(
    monitor='val_loss', # Metric to monitor
    patience=3,         # Number of epochs with no improvement after which training will be stopped.
    restore_best_weights=True, # Restores model weights from the epoch with the best value of the monitored metric.
    verbose=1           # Prints message when early stopping is triggered
)

print("\nTraining the NeuMF model...")
history = model.fit(
    [user_input_train, book_input_train, user_input_train, book_input_train],
    y_train,
    batch_size=64,
    epochs=10, # Can set a high number here, early stopping will manage it
    validation_data=([user_input_val, book_input_val, user_input_val, book_input_val], y_val),
    callbacks=[early_stopping_callback], # <--- Pass the callback here
    verbose=1
)

# Evaluate the model on the cross-validation set
print("\nEvaluating the model on the cross-validation set (disjoint users):")
loss, mae = model.evaluate([user_input_val, book_input_val, user_input_val, book_input_val], y_val, verbose=0)
print(f"Cross-Validation Loss (MSE): {loss:.4f}")
print(f"Cross-Validation MAE: {mae:.4f}")

# Plot training history (loss and MAE)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 4. Making Recommendations (Example for a single user) ---

def recommend_for_user(user_original_id, model, df_combined, user_id_to_idx, book_id_to_idx, idx_to_book_id, n_recommendations=10):
    """
    Generates top N book recommendations for a given user.

    Args:
        user_original_id: The original ID of the user for whom to generate recommendations.
        model: The trained Keras NeuMF model.
        df_combined: The combined DataFrame containing all user_id, book_id, and rating data (actual train + validation).
                     This is used to determine what books a user has already rated.
        user_id_to_idx: Dictionary mapping original user IDs to their integer indices.
        book_id_to_idx: Dictionary mapping original book IDs to their integer indices.
        idx_to_book_id: Dictionary mapping integer book indices back to original book IDs.
        n_recommendations: The number of top recommendations to return.

    Returns:
        A list of original book IDs recommended for the user.
    """
    if user_original_id not in user_id_to_idx:
        print(f"Error: User ID '{user_original_id}' not found in the dataset.")
        return []

    target_user_idx = user_id_to_idx[user_original_id]

    # Get books the user has already rated from the combined dataset using original IDs
    rated_book_original_ids = df_combined[df_combined['user_id'] == user_original_id]['book_id'].unique()

    # Convert rated original book IDs to their mapped indices
    rated_book_indices = [book_id_to_idx[book_id] for book_id in rated_book_original_ids if book_id in book_id_to_idx]

    # Get all unique book indices that exist in our mapping
    all_book_indices = np.array(list(idx_to_book_id.keys()))

    # Identify unrated books by comparing all book indices with rated book indices
    unrated_book_indices = np.setdiff1d(all_book_indices, rated_book_indices)

    if len(unrated_book_indices) == 0:
        print(f"User {user_original_id} has rated all available books or there are no unrated books to recommend.")
        return []

    # Prepare input for prediction for all unrated books
    # The user input will be the same user_idx repeated for each unrated book
    user_input_for_prediction = np.full(len(unrated_book_indices), target_user_idx)

    # Make predictions for the user across all unrated books
    predicted_ratings = model.predict([
        user_input_for_prediction, # User IDs for GMF
        unrated_book_indices,      # Book IDs for GMF
        user_input_for_prediction, # User IDs for MLP
        unrated_book_indices       # Book IDs for MLP
    ], verbose=0).flatten() # Suppress prediction progress output and flatten to 1D array

    # Create a DataFrame to store book indices and their predicted ratings
    recommendations_df = pd.DataFrame({
        'book_idx': unrated_book_indices,
        'predicted_rating': predicted_ratings
    })

    # Sort by predicted rating in descending order
    recommendations_df = recommendations_df.sort_values(by='predicted_rating', ascending=False)

    # Get the top N recommended book indices
    top_n_recommended_indices = recommendations_df['book_idx'].head(n_recommendations).tolist()

    # Convert recommended book indices back to their original IDs
    recommended_book_original_ids = [idx_to_book_id[idx] for idx in top_n_recommended_indices]

    return recommended_book_original_ids

# Example usage for recommendation:
# Combine actual training and validation data for the recommendation function to know all rated items
df_combined = pd.concat([df_train_actual, df_val], ignore_index=True)

# Pick a target user. This will try to get the first user ID from your loaded data.
if not df_combined['user_id'].empty:
    target_user_id_for_rec = df_combined['user_id'].iloc[0]
else:
    target_user_id_for_rec = None # No users available if data is empty

if target_user_id_for_rec:
    print(f"\nAttempting to generate recommendations for user: '{target_user_id_for_rec}'")
    top_recommendations = recommend_for_user(
        target_user_id_for_rec,
        model,
        df_combined, # Pass the combined data for recommendations
        user_id_to_idx,
        book_id_to_idx,
        idx_to_book_id,
        n_recommendations=10 # You can change this number
    )

    if top_recommendations:
        print(f"Top 10 recommended books for user '{target_user_id_for_rec}':")
        for i, book_id in enumerate(top_recommendations):
            print(f"{i+1}. {book_id}")
    else:
        print(f"Could not generate recommendations for user '{target_user_id_for_rec}'.")
else:
    print("\nSkipping recommendation generation as no user data was loaded.")


# --- 5. Handling New Users with Minimal Retraining (Fine-tuning) ---

def fine_tune_for_new_user(
    current_model: tf.keras.Model,
    df_full_train: pd.DataFrame,
    new_user_initial_ratings: pd.DataFrame,
    embedding_dim: int,
    epochs: int = 2,
    batch_size: int = 16
):
    """
    Fine-tunes the NeuMF model to incorporate a new user's initial ratings
    with minimal retraining.

    Args:
        current_model: The currently trained Keras NeuMF model.
        df_full_train: The DataFrame containing all existing training data (before new user).
        new_user_initial_ratings: DataFrame with the new user's initial ratings
                                  (must have 'user_id', 'book_id', 'rating').
        embedding_dim: The embedding dimension used in the model.
        epochs: Number of epochs for fine-tuning. Typically very small (e.g., 1-5).
        batch_size: Batch size for fine-tuning.

    Returns:
        tuple: (fine_tuned_model, updated_user_id_to_idx, updated_book_id_to_idx,
                updated_idx_to_user_id, updated_idx_to_book_id)
                Returns None if fine-tuning fails.
    """
    new_user_id_str = new_user_initial_ratings['user_id'].iloc[0]
    print(f"\n--- Fine-tuning for new user: {new_user_id_str} ---")

    # Combine new user data with the full training data to update overall mappings
    df_full_with_new_user = pd.concat([df_full_train, new_user_initial_ratings], ignore_index=True)

    # Re-create mappings with the new user included (important for correct n_users/n_books)
    new_all_users_category = df_full_with_new_user['user_id'].astype('category')
    new_all_books_category = df_full_with_new_user['book_id'].astype('category')

    new_all_users_categories = new_all_users_category.cat.categories
    new_all_books_categories = new_all_books_category.cat.categories

    updated_user_id_to_idx = {original_id: idx for idx, original_id in enumerate(new_all_users_categories)}
    updated_book_id_to_idx = {original_id: idx for idx, original_id in enumerate(new_all_books_categories)}
    updated_idx_to_user_id = {idx: original_id for original_id, idx in updated_user_id_to_idx.items()}
    updated_idx_to_book_id = {idx: original_id for original_id, idx in updated_book_id_to_idx.items()}

    new_n_users = len(new_all_users_categories)
    new_n_books = len(new_all_books_categories)

    print(f"Updated total unique users: {new_n_users}")
    print(f"Updated total unique books: {new_n_books}")

    try:
        # 1. Get original model's weights
        original_weights_by_name = {weight.name: weight.numpy() for weight in current_model.weights}

        # 2. Create a new model instance with potentially updated dimensions
        new_model = create_neumf_model(new_n_users, new_n_books, embedding_dim)

        # 3. Manually transfer weights
        for layer in new_model.layers:
            if layer.name.startswith('user_embedding_') or layer.name.startswith('book_embedding_'):
                # This is an embedding layer
                weight_name = layer.weights[0].name # Get the name of the embedding kernel weight
                if weight_name in original_weights_by_name:
                    original_embedding_weights = original_weights_by_name[weight_name]
                    
                    new_embedding_weights_var = layer.weights[0]
                    new_embedding_weights_array = new_embedding_weights_var.numpy()

                    # Determine the size to copy (min of original and new vocab size)
                    vocab_size_to_copy = min(original_embedding_weights.shape[0], new_embedding_weights_array.shape[0])
                    
                    # Copy the old embeddings into the new, larger matrix
                    new_embedding_weights_array[:vocab_size_to_copy, :] = original_embedding_weights[:vocab_size_to_copy, :]

                    # Assign the modified weights back to the layer
                    new_embedding_weights_var.assign(new_embedding_weights_array)
                else:
                    print(f"Warning: Original weights for embedding layer '{layer.name}' not found. Keeping random initialization.")
            elif layer.weights: # For other layers (like Dense layers)
                layer_weights_to_set = []
                all_found = True
                for weight_var in layer.weights:
                    if weight_var.name in original_weights_by_name:
                        layer_weights_to_set.append(original_weights_by_name[weight_var.name])
                    else:
                        all_found = False
                        print(f"Warning: Weight '{weight_var.name}' for layer '{layer.name}' not found in original model. Skipping transfer for this layer.")
                        break # Skip setting weights for this layer if any are missing
                if all_found:
                    try:
                        layer.set_weights(layer_weights_to_set)
                    except ValueError as ve:
                        print(f"Warning: Could not set weights for layer '{layer.name}' due to shape mismatch (possible in rare cases). Error: {ve}")


        print("Original weights loaded into the new model instance.")

        # Compile the new model for fine-tuning
        new_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        # Prepare new user's data for fine-tuning
        fine_tune_data_mapped = new_user_initial_ratings.copy()
        fine_tune_data_mapped['user_idx'] = fine_tune_data_mapped['user_id'].map(updated_user_id_to_idx)
        fine_tune_data_mapped['book_idx'] = fine_tune_data_mapped['book_id'].map(updated_book_id_to_idx)

        new_user_input_fine_tune = fine_tune_data_mapped['user_idx'].values.astype(int)
        new_book_input_fine_tune = fine_tune_data_mapped['book_idx'].values.astype(int)
        new_y_fine_tune = fine_tune_data_mapped['rating'].values

        print(f"Fine-tuning model for {new_user_id_str} on {len(new_user_input_fine_tune)} initial ratings for {epochs} epochs...")
        new_model.fit(
            [new_user_input_fine_tune, new_book_input_fine_tune, new_user_input_fine_tune, new_book_input_fine_tune],
            new_y_fine_tune,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0 # Suppress verbose output during fine-tuning
        )
        print("Fine-tuning complete.")

        return new_model, updated_user_id_to_idx, updated_book_id_to_idx, updated_idx_to_user_id, updated_idx_to_book_id

    except Exception as e:
        print(f"Error during new user fine-tuning: {e}")
        return None, None, None, None, None

# --- Example Usage for New User Handling ---
if __name__ == '__main__':
    # Define a hypothetical new user's initial ratings based on the provided data
    new_user_id_example = 'new_user_provided_data'
    sample_user_books = [
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

    sample_user_ratings = [5, 5, 5, 3, 4, 5, 2, 4, 5, 4, 5, 4, 4, 5, 4, 5, 4]

    example_new_user_ratings = pd.DataFrame({
        'user_id': [new_user_id_example] * len(sample_user_books),
        'book_id': sample_user_books,
        'rating': sample_user_ratings
    })

    # Fine-tune the model for the new user
    fine_tuned_model, updated_user_id_to_idx, updated_book_id_to_idx, updated_idx_to_user_id, updated_idx_to_book_id = \
        fine_tune_for_new_user(
            model, # Pass the initially trained model
            df_full_train, # Pass the original full training data
            example_new_user_ratings,
            embedding_dim=embedding_dim,
            epochs=2, # Small number of epochs for fine-tuning
            batch_size=len(example_new_user_ratings) # Fine-tune on the whole new user data as one batch
        )

    if fine_tuned_model:
        # Now generate recommendations for the new user using the fine-tuned model
        # Create a combined dataset for recommendations that includes the new user's initial ratings
        df_combined_for_new_user_rec = pd.concat([df_combined, example_new_user_ratings], ignore_index=True)

        print(f"\nAttempting to generate recommendations for NEW user: '{new_user_id_example}'")
        # Call recommend_for_user with the fine-tuned model and updated mappings
        top_recommendations_new_user = recommend_for_user(
            new_user_id_example,
            fine_tuned_model, # Use the fine-tuned model
            df_combined_for_new_user_rec, # Use the expanded combined data
            updated_user_id_to_idx, # Use the updated mappings
            updated_book_id_to_idx,
            updated_idx_to_book_id,
            n_recommendations=20 # Changed to 20 recommendations
        )

        if top_recommendations_new_user:
            print(f"Top {len(top_recommendations_new_user)} recommended books for new user '{new_user_id_example}':")
            for i, book_id in enumerate(top_recommendations_new_user):
                print(f"{i+1}. {book_id}")
        else:
            print(f"Could not generate recommendations for new user '{new_user_id_example}'.")
    else:
        print("Fine-tuning for new user failed. Skipping new user recommendations.")
