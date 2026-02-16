import numpy as np
import matplotlib.pyplot as plt
from mlcore.linear_models.linear_regression import LinearRegression

def run_experiment():
    X = np.random.randn(100, 10)
    y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.5
    alphas = [0, 0.01, 0.1, 1, 10, 100]
    weights = []

    for a in alphas:
        model = LinearRegression(lr=0.01, n_iters=2000, alpha=a, l1_ratio=0.5)
        model.fit(X, y)
        weights.append(model.w)
    weights = np.array(weights)
    
    plt.figure(figsize=(10, 6))
    for i in range(X.shape[1]):
        plt.plot(alphas, weights[:, i], label=f'Feature {i}', marker='o')
    
    plt.xscale('log')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('Weight Magnitude')
    plt.title('Weight Shrinkage: Effect of Regularization (Lasso/Ridge)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig('reports/figures/regularization.png')
    plt.show()

if __name__ == "__main__":
    run_experiment()