import time
import numpy as np
from sklearn.linear_model import LogisticRegression as SKLogistic
from sklearn.metrics import accuracy_score
from mlcore.linear_models.logistic_regression import LogisticRegression as MyLogistic

def benchmark():
    X = np.random.randn(1000, 20)
    y = (np.dot(X, np.random.randn(20)) > 0).astype(int)
    start = time.time()
    model_custom = MyLogistic(lr=0.1, n_iters=1000, alpha=0.01)
    model_custom.fit(X, y)
    custom_time = time.time() - start
    custom_acc = accuracy_score(y, model_custom.predict(X))

    start = time.time()
    model_sk = SKLogistic(penalty='l2', C=100.0, solver='lbfgs')
    model_sk.fit(X, y)
    sk_time = time.time() - start
    sk_acc = accuracy_score(y, model_sk.predict(X))

    print(f"{'Metric':<15} | {'Custom':<10} | {'Sklearn':<10}")
    print("-" * 45)
    print(f"{'Accuracy':<15} | {custom_acc:<10.4f} | {sk_acc:<10.4f}")
    print(f"{'Time (s)':<15} | {custom_time:<10.4f} | {sk_time:<10.4f}")
    
    diff = np.abs(custom_acc - sk_acc)
    print(f"\nParity Check: {'PASSED' if diff < 0.05 else 'FAILED'} (Diff: {diff:.4f})")

if __name__ == "__main__":
    benchmark()