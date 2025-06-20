import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

ratings = pd.read_csv('../fake_data/ratings.csv')

ratings["book"] = ratings[['Title', 'Author']].agg(', '.join, axis=1)

ratings = ratings[["book","User_id","Rating"]]

ratings = ratings[ratings['Rating'] != 'Invalid rating']
ratings['Rating'] = ratings['Rating'].astype(int)


pivot_ratings = ratings.pivot_table(index='book', columns='User_id', values='Rating')

#pivot_ratings.to_csv('../fake_data/user_array.csv')


Y = pivot_ratings.to_numpy()

R = np.zeros_like(Y, dtype=int)
non_nan_mask = ~np.isnan(Y)
R[non_nan_mask] = 1



num_books, num_users = Y.shape
num_features = 200

###


book_list = pivot_ratings.index.tolist()


test_ratings = Y[:,0]


test_rated = [i for i in range(len(test_ratings)) if test_ratings[i] > 0]

print('\nTest user ratings:\n')
for i in range(len(test_ratings)):
    if test_ratings[i] > 0 :
        print(f'Rated {test_ratings[i]} for  {book_list[i]}')





####


Ymean = np.nanmean(Y,axis=1)
Ynorm = (Y.T-Ymean).T



# # ####



def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Vectorized for speed. Uses tensorflow operations to be compatible with custom training loop.
    Args:
      X (ndarray (num_books,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_books,num_users)    : matrix of user ratings of books
      R (ndarray (num_books,num_users)    : matrix, where R(i, j) = 1 if the i-th books was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    Y = np.nan_to_num(Y)
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J




# Set Initial Parameters (W, X), use tf.Variable to track these variables
W = tf.Variable(tf.random.normal((num_users,  num_features),dtype=tf.float64),  name='W')
X = tf.Variable(tf.random.normal((num_books, num_features),dtype=tf.float64),  name='X')
b = tf.Variable(tf.random.normal((1, num_users),dtype=tf.float64),  name='b')



#Instantiate an optimizer.
optimizer = keras.optimizers.Adam(learning_rate=1e-1)

iterations = 200
lambda_ = 1
for iter in range(iterations):
    # Use TensorFlow’s GradientTape
    # to record the operations used to compute the cost 
    with tf.GradientTape() as tape:

        # Compute the cost (forward pass included in cost)
        cost_value = cofi_cost_func_v(X, W, b, Ynorm, R, lambda_)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss
    grads = tape.gradient( cost_value, [X,W,b] )

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients( zip(grads, [X,W,b]) )

    # Log periodically.
    if iter % 20 == 0:
        print(f"Training loss at iteration {iter}: {cost_value:0.1f}")


# Make a prediction using trained weights and biases
p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()

#restore the mean
pm = (p.T + Ymean).T

my_predictions = pm[:,0]

# sort predictions
ix = tf.argsort(my_predictions, direction='DESCENDING')

for i in range(17):
    j = ix[i]
    if j not in test_rated:
        print(f'Predicting rating {my_predictions[j]:0.2f} for book {book_list[j]}')

print('\n\nOriginal vs Predicted ratings:\n')
for i in range(len(test_ratings)):
    if test_ratings[i] > 0:
        print(f'Original {test_ratings[i]}, Predicted {my_predictions[i]:0.2f} for {book_list[i]}')
