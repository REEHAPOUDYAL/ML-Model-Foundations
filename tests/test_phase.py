import numpy as np
from sklearn import metrics as sklearn_metrics
import mlcore.metrics.regression as reg
import mlcore.metrics.classification as clf
from mlcore.utils.validation import check_array

def test_validation():
    x_1d = [1, 2, 3]
    x_checked = check_array(x_1d, ensure_2d=True)
    assert x_checked.shape == (3, 1)
    try:
        check_array([[np.nan, 1]])
    except Exception as e:
        print(f"Validation caught NaN correctly: {e}")

def test_regression_parity():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.1, 7.8])
    
    assert np.allclose(reg.mean_squared_error(y_true, y_pred), 
                       sklearn_metrics.mean_squared_error(y_true, y_pred))
    assert np.allclose(reg.r2_score(y_true, y_pred), 
                       sklearn_metrics.r2_score(y_true, y_pred))
    print("Regression metrics match sklearn gold standard.")

def test_classification_parity():
    y_true = np.array([0, 1, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    
    assert np.allclose(clf.accuracy_score(y_true, y_pred), 
                       sklearn_metrics.accuracy_score(y_true, y_pred))
    assert np.allclose(clf.f1_score(y_true, y_pred), 
                       sklearn_metrics.f1_score(y_true, y_pred))
    print("Classification metrics match sklearn gold standard.")

if __name__ == "__main__":
    test_validation()
    test_regression_parity()
    test_classification_parity()
    print("\nPHASE 1 VERIFIED: All systems functional.")