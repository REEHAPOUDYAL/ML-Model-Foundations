import numpy as np
import pytest
from mlcore.linear_models.logistic_regression import LogisticRegression

def test_logistic_convergence():
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    model = LogisticRegression(lr=0.1, n_iters=1000)
    model.fit(X, y)    
    preds = model.predict(X)
    assert np.array_equal(preds, y)

def test_sigmoid_stability():
    model = LogisticRegression()
    large_z = np.array([1000.0, -1000.0, 0.0])
    probs = model._sigmoid(large_z)
    
    assert probs[0] == 1.0
    assert probs[1] == 0.0
    assert probs[2] == 0.5
    assert not np.isnan(probs).any()

def test_logistic_regularization():
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)    
    model_no_reg = LogisticRegression(lr=0.1, n_iters=500, alpha=0)
    model_reg = LogisticRegression(lr=0.1, n_iters=500, alpha=10.0)
    model_no_reg.fit(X, y)
    model_reg.fit(X, y)
    assert np.linalg.norm(model_reg.w) < np.linalg.norm(model_no_reg.w)

def test_probability_outputs():
    X = np.random.randn(10, 2)
    y = np.random.randint(0, 2, 10)
    model = LogisticRegression()
    model.fit(X, y)
    
    probs = model.predict_proba(X)
    assert np.all((probs >= 0) & (probs <= 1))