"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    
    X_copy = np.array(X, dtype = float)

    #Check where is NaN
    mask = np.isnan(X_copy)

    # Calculate means and clean NaN means by turning them to 0
    means = np.nanmean(X_copy, axis = 0)
    means = np.where(np.isnan(means),0.0 , means)

    # Where X is NaN fill it with means if not keep the original
    X_copy = np.where(mask == True, means, X_copy)

    return X_copy

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):

    # Calculate quartiles and IQR
    q1 = np.percentile(X, 25, axis = 0) 
    q3 = np.percentile(X, 75, axis = 0) 
    iqr = q3 - q1

    # Calculate bounds

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    return lower, upper

# Step 3 - clip_columns
def clip_columns(X, lower, upper):

    # Clip the data to the lower and upper limits
    return np.clip(X, lower, upper)

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    
    return numerator / (denominator + eps)

# Step 5 - append_column
def append_column(X, col):
    
    return np.column_stack([X, col])

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    
    # Geting unique labels
    uniques, inverse = np.unique(labels, return_inverse = True)

    # Zeros matrix
    zeros = np.zeros((len(labels), len(uniques))).astype(float)

    # Encoding
    zeros[np.arange(len(labels)), inverse] = 1.0

    return zeros

# Step 7 - fit_standardizer
def fit_standardizer(X):

    mean = np.mean(X, axis = 0)

    std = np.std(X, axis = 0)
    std = np.where(std == 0, 1.0, std)

    return mean, std

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    
    return (X - mean) / std

# Step 9 - add_bias_column
def add_bias_column(X):

    ones = np.ones((X.shape[0], 1))

    return np.hstack([ones, X])

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):

    rng = np.random.default_rng(seed)
    
    return rng.permutation(n_samples)

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):

    n = len(indices)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[: n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val : ]

    return train_idx, val_idx, test_idx

# Step 12 - subset_xy
def subset_xy(X, y, indices):

    X_sub = X[indices]
    y_sub = y[indices]

    return X_sub, y_sub

# Step 13 - ols_fit
def ols_fit(X, y):
    
    # Form the Gram matrix
    A = X.T @ X

    # Right hand side
    b = X.T @ y

    return np.linalg.solve(A, b)

# Step 14 - ols_predict
def ols_predict(X, theta):

    return np.dot(X, theta)

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    
    return float(np.mean(np.abs(y_true - y_pred)))

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

# Step 17 - r_squared
def r_squared(y_true, y_pred):

    y_mean = np.mean(y_true)

    ssres = np.sum((y_true - y_pred)**2)
    sstot = np.sum((y_true - y_mean)**2)

    return float(1 - ssres/sstot) if sstot != 0 else 0.0

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    
    assert y_true.shape == y_pred.shape

    r = y_true - y_pred
    mean = float(np.mean(r))
    std = float(np.std(r))
    median_abs = float(np.median(np.abs(r)))

    return {
        'mean' : mean,
        'std' : std,
        'median_abs' : median_abs
    }

# Step 19 - prepare_cleaned_features (not yet solved)
# TODO: implement

# Step 20 - assemble_feature_matrix (not yet solved)
# TODO: implement

# Step 21 - make_train_val_test (not yet solved)
# TODO: implement

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

