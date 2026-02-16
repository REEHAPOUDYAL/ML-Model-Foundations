import numpy as np
from .validation import check_X_y

def train_test_split(X, y, test_size=0.2, random_state=None):
    X, y = check_X_y(X, y)
    
    if random_state:
        np.random.seed(random_state)
    
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    split_idx = int(len(indices) * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]