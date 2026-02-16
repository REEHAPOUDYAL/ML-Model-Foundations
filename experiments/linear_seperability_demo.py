import numpy as np
import matplotlib.pyplot as plt
from mlcore.linear_models.logistic_regression import LogisticRegression
from mlcore.svm.linear_svm import LinearSVM # Ensure this exists or use Logistic

def run_separability_study():
    np.random.seed(42)
    
    X_lin = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
    y_lin = np.array([0]*20 + [1]*20)
    X_xor = np.random.randn(40, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0).astype(int)

    datasets = [(X_lin, y_lin, "Linearly Separable"), (X_xor, y_xor, "XOR (Non-Linear)")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, (X, y, title) in zip(axes, datasets):
        model = LogisticRegression(lr=0.1, n_iters=1000)
        model.fit(X, y)
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='RdBu')
        ax.set_title(f"{title}\nAccuracy: {np.mean(model.predict(X) == y):.2f}")

    plt.tight_layout()
    plt.savefig('reports/figures/separability.png')
    plt.show()

if __name__ == "__main__":
    run_separability_study()