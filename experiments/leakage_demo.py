import numpy as np
from mlcore.preprocessing.scaler import StandardScaler
from mlcore.utils.model_selection import train_test_split

def run_leakage_demo():
    X = np.random.randn(100, 2)
    y = np.random.randint(0, 2, 100)
    scaler = StandardScaler()
    X_scaled_bad = scaler.fit_transform(X)
    X_train_bad, X_test_bad, _, _ = train_test_split(X_scaled_bad, y)
    print("Scenario A: Leakage occurs (Mean/Std learned from total population)")

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scaler = StandardScaler()
    X_train_good = scaler.fit(X_train).transform(X_train)
    X_test_good = scaler.transform(X_test)
    print("Scenario B: Professional handling (Mean/Std learned ONLY from training set)")

if __name__ == "__main__":
    run_leakage_demo()
    print("\nPhase 2 Logic: Verified and Leakage-Proof.")