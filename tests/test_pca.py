import numpy as np
from sklearn.decomposition import PCA as SklearnPCA
from mlcore.decomposition.pca import PCA

def test_pca_parity():
    np.random.seed(42)
    X = np.random.rand(100, 5)
    my_pca = PCA(n_components=2)
    X_transformed_my = my_pca.fit_transform(X)
    sk_pca = SklearnPCA(n_components=2)
    X_transformed_sk = sk_pca.fit_transform(X)
    assert X_transformed_my.shape == (100, 2)
    
    for i in range(my_pca.n_components):
        dot_product = np.abs(np.dot(my_pca.components[i], sk_pca.components_[i]))
        assert np.allclose(dot_product, 1.0, atol=1e-5)
        
    print("PCA parity check passed.")

def test_variance_explanation():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    pca = PCA(n_components=1)
    pca.fit(X)
    assert pca.explained_variance_ is not None
    assert len(pca.explained_variance_) == 1
    print("PCA variance explanation verified.")

if __name__ == "__main__":
    test_pca_parity()
    test_variance_explanation()
    print("\nPHASE 5 STRUCTURAL ANALYSIS VERIFIED.")