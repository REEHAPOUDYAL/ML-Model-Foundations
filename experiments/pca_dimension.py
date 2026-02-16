import numpy as np
import matplotlib.pyplot as plt
from mlcore.decomposition.pca import PCA

def run_pca_demo():
    np.random.seed(42)
    mean = [0, 0, 0]
    cov = [[1, 0.8, 0.8], [0.8, 1, 0.8], [0.8, 0.8, 1]]
    X = np.random.multivariate_normal(mean, cov, 100)

    pca = PCA(n_components=2)
    pca.fit(X)
    X_projected = pca.transform(X)

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Ratio')
    plt.title('Explained Variance')
    plt.subplot(1, 2, 2)
    plt.scatter(X_projected[:, 0], X_projected[:, 1], alpha=0.7, edgecolors='k')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('2D Projection of 3D Data')
    plt.tight_layout()
    plt.savefig('reports/figures/pca_variance.png')
    plt.show()

if __name__ == "__main__":
    run_pca_demo()