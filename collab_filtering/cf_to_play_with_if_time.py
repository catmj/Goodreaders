import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dot, Concatenate, Dense, Flatten
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# --- 1. Data Preparation ---
# Assume 'ratings.csv' has 'user_id', 'book_id', 'rating' columns
# Replace with your actual data loading
data = pd.read_csv('ratings.csv')

# Map IDs to contiguous integers
user_ids = data['user_id'].astype('category').cat.codes.values
book_ids = data['book_id'].astype('category').cat.codes.values
ratings = data['rating'].values

n_users = len(np.unique(user_ids))
n_books = len(np.unique(book_ids))
embedding_dim = 50 # Hyperparameter

# Split data
X = np.stack([user_ids, book_ids], axis=1)
y = ratings
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

user_input_train, book_input_train = X_train[:, 0], X_train[:, 1]
user_input_test, book_input_test = X_test[:, 0], X_test[:, 1]

# --- 2. Model Architecture (NeuMF - Neural Matrix Factorization) ---

# GMF Branch
user_input_gmf = Input(shape=(1,), name='user_input_gmf')
book_input_gmf = Input(shape=(1,), name='book_input_gmf')

user_embedding_gmf = Embedding(n_users, embedding_dim, name='user_embedding_gmf')(user_input_gmf)
book_embedding_gmf = Embedding(n_books, embedding_dim, name='book_embedding_gmf')(book_input_gmf)

user_vec_gmf = Flatten()(user_embedding_gmf)
book_vec_gmf = Flatten()(book_embedding_gmf)

# Element-wise product
gmf_output = Dot(axes=1)([user_vec_gmf, book_vec_gmf])


# MLP Branch
user_input_mlp = Input(shape=(1,), name='user_input_mlp')
book_input_mlp = Input(shape=(1,), name='book_input_mlp')

user_embedding_mlp = Embedding(n_users, embedding_dim, name='user_embedding_mlp')(user_input_mlp)
book_embedding_mlp = Embedding(n_books, embedding_dim, name='book_embedding_mlp')(book_input_mlp)

user_vec_mlp = Flatten()(user_embedding_mlp)
book_vec_mlp = Flatten()(book_embedding_mlp)

# Concatenate embeddings
concat_mlp = Concatenate()([user_vec_mlp, book_vec_mlp])

# MLP layers
mlp_output = Dense(128, activation='relu')(concat_mlp)
mlp_output = Dense(64, activation='relu')(mlp_output)
mlp_output = Dense(32, activation='relu')(mlp_output) # You can add more layers


# Combine GMF and MLP outputs
# Note: In the original NeuMF paper, they combine the last hidden layers, not the final output.
# For simplicity here, let's combine their direct outputs for demonstration.
# For a more faithful implementation, combine the pre-output layers of GMF and MLP.
combined_output = Concatenate()([gmf_output, mlp_output])

# Final prediction layer
# For rating prediction (regression), use a linear activation
prediction = Dense(1, activation='linear')(combined_output)

# Create the model
model = Model(inputs=[user_input_gmf, book_input_gmf, user_input_mlp, book_input_mlp], outputs=prediction)

# --- 3. Training the Model ---
model.compile(optimizer='adam', loss='mean_squared_error') # MSE for regression

model.summary()

# Train the model
history = model.fit(
    [user_input_train, book_input_train, user_input_train, book_input_train], # Pass inputs for both branches
    y_train,
    batch_size=64,
    epochs=10, # Adjust as needed
    validation_data=([user_input_test, book_input_test, user_input_test, book_input_test], y_test)
)

# Evaluate on test set
loss = model.evaluate([user_input_test, book_input_test, user_input_test, book_input_test], y_test)
print(f"Test Loss (MSE): {loss}")

# --- 4. Making Recommendations (Example for a single user) ---

# Get user ID mapping (assuming you saved it)
unique_users = data['user_id'].astype('category').cat.categories
unique_books = data['book_id'].astype('category').cat.categories

user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
book_to_idx = {book: idx for idx, book in enumerate(unique_books)}

# Let's say we want recommendations for user 'UserX' (replace with an actual user ID)
target_user_id = unique_users[0] # Example: first user in your data
target_user_idx = user_to_idx[target_user_id]

# Get books the user has already rated
rated_book_ids = data[data['user_id'] == target_user_id]['book_id'].values
rated_book_indices = [book_to_idx[book_id] for book_id in rated_book_ids]

# Get all book indices
all_book_indices = np.arange(n_books)

# Identify unrated books
unrated_book_indices = np.setdiff1d(all_book_indices, rated_book_indices)

# Prepare input for prediction
user_input_for_prediction = np.full(len(unrated_book_indices), target_user_idx)

predicted_ratings = model.predict([
    user_input_for_prediction,
    unrated_book_indices,
    user_input_for_prediction,
    unrated_book_indices
]).flatten()

# Create a DataFrame for easier sorting and lookup
recommendations_df = pd.DataFrame({
    'book_idx': unrated_book_indices,
    'predicted_rating': predicted_ratings
})

# Sort by predicted rating
recommendations_df = recommendations_df.sort_values(by='predicted_rating', ascending=False)

# Get original book IDs
recommended_book_original_ids = [unique_books[idx] for idx in recommendations_df['book_idx'].head(10)]

print(f"\nTop 10 recommended books for user {target_user_id}:")
print(recommended_book_original_ids)