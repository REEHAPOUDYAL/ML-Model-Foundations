import numpy as np
import matplotlib.pyplot as plt
from mlcore.linear_models.linear_regression import LinearRegression
from mlcore.metrics.regression import mean_squared_error
from mlcore.preprocessing.scaler import StandardScaler

def run_bias_variance_study():
    np.random.seed(42)
    X = np.sort(np.random.uniform(-3, 3, (100, 1)), axis=0)
    y = 0.5 * X**3 + 0.5 * X**2 + X + 2 + np.random.normal(0, 1.0, (100, 1))
    
    X_poly = np.hstack([X**i for i in range(1, 6)])
    
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]
    
    X_train_raw, X_test_raw = X_poly[train_idx], X_poly[test_idx]
    y_train, y_test = y[train_idx].ravel(), y[test_idx].ravel()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    alphas = np.logspace(-5, 2, 25)
    train_errors, test_errors = [], []

    for a in alphas:
        model = LinearRegression(lr=0.01, n_iters=2000, alpha=a, l1_ratio=0.0)
        model.fit(X_train, y_train)
        
        train_errors.append(mean_squared_error(y_train, model.predict(X_train)))
        test_errors.append(mean_squared_error(y_test, model.predict(X_test)))

    plt.figure(figsize=(10, 6))
    plt.loglog(alphas, train_errors, 'b-o', label='Train Error (Bias)', markersize=4)
    plt.loglog(alphas, test_errors, 'r-s', label='Test Error (Variance)', markersize=4)
    
    plt.xlabel('Regularization Strength (Alpha)')
    plt.ylabel('MSE (Log Scale)')
    plt.title('Bias-Variance Tradeoff: Impact of Regularization')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_bias_variance_study()