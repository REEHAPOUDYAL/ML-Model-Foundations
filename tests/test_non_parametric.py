import numpy as np
from sklearn import datasets
from sklearn import neighbors, naive_bayes, svm
from sklearn.metrics import accuracy_score as sk_accuracy

from mlcore.neighbors.knn import KNeighborsClassifier
from mlcore.bayes.naive_bayes import GaussianNB
from mlcore.svm.linear_svm import LinearSVC
from mlcore.preprocessing.scaler import StandardScaler

def test_knn_parity():
    X, y = datasets.make_classification(n_samples=50, n_features=4, random_state=42)
    
    sk_model = neighbors.KNeighborsClassifier(n_neighbors=3)
    sk_model.fit(X, y)
    sk_preds = sk_model.predict(X)
    
    my_model = KNeighborsClassifier(k=3)
    my_model.fit(X, y)
    my_preds = my_model.predict(X)
    
    assert np.mean(my_preds == sk_preds) >= 0.95
    print("KNN parity check passed.")

def test_nb_parity():
    X, y = datasets.make_classification(n_samples=50, n_features=4, random_state=42)
    
    sk_model = naive_bayes.GaussianNB()
    sk_model.fit(X, y)
    sk_preds = sk_model.predict(X)
    
    my_model = GaussianNB()
    my_model.fit(X, y)
    my_preds = my_model.predict(X)
    
    assert np.mean(my_preds == sk_preds) >= 0.95
    print("Naive Bayes parity check passed.")

def test_svm_logic():
    X, y = datasets.make_blobs(n_samples=50, centers=2, random_state=6)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    my_model = LinearSVC(lr=0.001, lambda_param=0.01, n_iter=1000)
    my_model.fit(X, y)
    my_preds = my_model.predict(X)
    
    accuracy = np.mean(my_preds == y)
    assert accuracy > 0.8
    print(f"SVM Logic check passed with accuracy: {accuracy}")

if __name__ == "__main__":
    test_knn_parity()
    test_nb_parity()
    test_svm_logic()
    print("\n phase 4 sucessful")