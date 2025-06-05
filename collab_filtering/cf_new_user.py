import tensorflow as tf
import numpy as np
from tensorflow import keras

# Assume X, Ynorm, R are already defined and are TensorFlow Tensors or NumPy arrays
# X: (num_books, num_features) - book features (fixed)
# Ynorm: (num_books, num_users) - normalized ratings (full dataset)
# R: (num_books, num_users) - rating indicator (full dataset)

# Let's create some dummy data for demonstration if you don't have it readily available
# In your actual code, X, Ynorm, R would come from your pre-existing model/data.
num_books = 100
num_users = 50
num_features = 10
lambda_ = 1

X = tf.Variable(np.random.rand(num_books, num_features), dtype=tf.float32, name='X_fixed', trainable=False) # X is now fixed
# Simulate existing W and b from a previous training
W_existing = tf.Variable(np.random.rand(num_users, num_features), dtype=tf.float32, name='W_existing', trainable=False)
b_existing = tf.Variable(np.random.rand(1, num_users), dtype=tf.float32, name='b_existing', trainable=False)

# Create dummy Ynorm and R
Ynorm = np.random.rand(num_books, num_users) * 5 # Ratings from 0 to 5
R = (Ynorm > 0).astype(int) # R = 1 if rated, 0 otherwise
Ynorm[R == 0] = np.nan # Introduce NaNs for unrated items
Ynorm = tf.constant(Ynorm, dtype=tf.float32)
R = tf.constant(R, dtype=tf.float32)

# --- Simulate New User Data ---
# For a new user, you'd have their ratings and the corresponding indicator matrix
# Let's assume the new user is index `new_user_idx` (or just one new user)
new_user_idx = 0 # For demonstration, let's pick the first user's data as 'new'
                  # In a real scenario, this would be new, unseen data.

# Extract the new user's ratings and R vector
Y_new_user_vector = Ynorm[:, new_user_idx:new_user_idx+1] # Shape (num_books, 1)
R_new_user_vector = R[:, new_user_idx:new_user_idx+1]     # Shape (num_books, 1)

# Important: If your new user has *no* ratings yet, `Y_new_user_vector` and `R_new_user_vector`
# would be all NaNs or zeros. You need to gather their *actual* initial ratings.
# For example:
# new_user_ratings_dict = {book_id_1: rating_1, book_id_2: rating_2}
# Y_new_user_vector_actual = np.zeros((num_books, 1))
# R_new_user_vector_actual = np.zeros((num_books, 1))
# for book_id, rating in new_user_ratings_dict.items():
#     Y_new_user_vector_actual[book_id, 0] = rating
#     R_new_user_vector_actual[book_id, 0] = 1
# Y_new_user_vector = tf.constant(Y_new_user_vector_actual, dtype=tf.float32)
# R_new_user_vector = tf.constant(R_new_user_vector_actual, dtype=tf.float32)


# --- Initialize parameters for the new user ---
# These are the only variables we want to train
W_new_user = tf.Variable(np.random.rand(1, num_features), dtype=tf.float32, name='W_new_user') # Shape (1, num_features)
b_new_user = tf.Variable(np.random.rand(1, 1), dtype=tf.float32, name='b_new_user')           # Shape (1, 1)


def cofi_cost_func_new_user(X_fixed, W_new_user, b_new_user, Y_new_user, R_new_user, lambda_):
    """
    Returns the cost for a *single new user*, keeping X fixed.
    Args:
      X_fixed (tf.Tensor (num_books,num_features)): matrix of item features (fixed)
      W_new_user (tf.Variable (1,num_features)) : latent factors for the new user
      b_new_user (tf.Variable (1, 1))           : bias for the new user
      Y_new_user (tf.Tensor (num_books, 1))     : new user's ratings
      R_new_user (tf.Tensor (num_books, 1))     : indicator matrix for new user's ratings
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    # Replace NaNs with zeros in Y_new_user for calculation purposes
    Y_new_user_filled = tf.where(tf.math.is_nan(Y_new_user), tf.zeros_like(Y_new_user), Y_new_user)

    # Prediction for the new user: (num_books, num_features) @ (num_features, 1) + (1, 1)
    # The result will be (num_books, 1)
    predictions = tf.linalg.matmul(X_fixed, tf.transpose(W_new_user)) + b_new_user

    # Error term, only for rated items by the new user
    j = (predictions - Y_new_user_filled) * R_new_user

    # Squared error for rated items
    error_squared = tf.reduce_sum(j**2)

    # Regularization terms
    # Note: X_fixed is fixed, so its regularization term is not relevant for *training* W_new_user/b_new_user.
    # We only regularize the *trainable* variables (W_new_user, b_new_user).
    # If you were calculating the total cost for the full system, you would include X_fixed's regularization.
    # But for optimizing only the new user's parameters, we only regularize those.
    regularization_W = tf.reduce_sum(W_new_user**2)
    # No regularization on bias (b) is common, but you could add it if desired.
    # regularization_b = tf.reduce_sum(b_new_user**2)

    J = 0.5 * error_squared + (lambda_/2) * regularization_W # + (lambda_/2) * regularization_b

    return J

# Instantiate an optimizer.
optimizer_new_user = keras.optimizers.Adam(learning_rate=1e-1)

iterations_new_user = 100 # Fewer iterations are usually sufficient for a single user
lambda_ = 1 # Keep the regularization parameter the same

print("Training parameters for the new user...")
for iter in range(iterations_new_user):
    # Use TensorFlow’s GradientTape
    # to record the operations used to compute the cost
    with tf.GradientTape() as tape:
        # Compute the cost for the new user
        cost_value_new_user = cofi_cost_func_new_user(
            X, # X is fixed
            W_new_user,
            b_new_user,
            Y_new_user_vector,
            R_new_user_vector,
            lambda_
        )

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables (W_new_user, b_new_user)
    grads_new_user = tape.gradient(cost_value_new_user, [W_new_user, b_new_user])

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer_new_user.apply_gradients(zip(grads_new_user, [W_new_user, b_new_user]))

    # Log periodically.
    if iter % 10 == 0:
        print(f"New User Training loss at iteration {iter}: {cost_value_new_user:0.4f}")

print("\nNew user parameters (W_new_user, b_new_user) trained.")
print(f"W_new_user shape: {W_new_user.shape}, value: {W_new_user.numpy()}")
print(f"b_new_user shape: {b_new_user.shape}, value: {b_new_user.numpy()}")


# --- How to use the new user's parameters for prediction ---
# Now, to predict the new user's preference for an unrated book (e.g., book_id = 5)
book_id_to_predict = 5
predicted_rating = tf.linalg.matmul(tf.expand_dims(X[book_id_to_predict], axis=0), tf.transpose(W_new_user)) + b_new_user
print(f"\nPredicted rating for book {book_id_to_predict} by new user: {predicted_rating.numpy()[0,0]:.2f}")
