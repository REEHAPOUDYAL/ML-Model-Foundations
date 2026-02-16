# Machine Learning Fundamentals (From Scratch)

A **from-scratch Machine Learning toolkit** implemented using **NumPy-based vectorization** to demonstrate strong understanding of **optimization, numerical stability, regularization, and model evaluation**, without relying on high-level ML frameworks like **scikit-learn**.

This project focuses on building core ML algorithms manually and validating them through experimental analysis and sklearn parity testing.

---

## Project Highlights

- Fully implemented ML models **without sklearn training utilities**
- Focus on **mathematical correctness** and **gradient-based optimization**
- Includes **regularization (L1/L2/ElasticNet)** and stability-safe loss computations
- Verified using **unit tests (pytest)** and **benchmarking vs sklearn**

---

## Implemented Algorithms

### 1. Linear Regression
- Implemented using **Batch Gradient Descent**
- Objective: minimize **Mean Squared Error (MSE)**
- Supports:
  - **L1 (Lasso)** regularization
  - **L2 (Ridge)** regularization
  - **Elastic Net** (L1 + L2)

---

### 2. Logistic Regression
- Implemented using **Batch Gradient Descent**
- Objective: minimize **Cross-Entropy Loss**
- Includes:
  - **stabilized sigmoid**
  - **log-clipping** to prevent overflow / underflow
  - ElasticNet regularization support

---

### 3. Support Vector Machine (SVM)
- Binary SVM implementation for classification
- Used to isolate **Iris Setosa** from UCI Iris dataset
- Uses soft-margin approach:
  - maximizes margin
  - penalizes misclassification using regularization parameter

---

### 4. Principal Component Analysis (PCA)
- Implemented using:
  - covariance matrix computation
  - eigen decomposition
- Outcome:
  - **PC1 captures ~80% variance** in tested datasets

---

### 5. K-Nearest Neighbors (KNN)
- Fully non-parametric classifier
- Efficient implementation using:
  - **vectorized Euclidean distance**
  - majority vote prediction logic

---

## Experimental Analysis

### Bias–Variance Tradeoff Experiment
- Performed **5th-degree polynomial regression**
- Trained on cubic signal
- Evaluated with varying regularization strengths
- Successfully produced the expected **U-shaped error curve**
- Identified best regularization region (underfitting → optimal → overfitting)

---

### Linear Separability Study (XOR)
- Compared performance on:
  - linearly separable dataset
  - XOR non-linear dataset
- Demonstrated limitation of linear classifiers on XOR-type distributions

---

## Validation and Testing

### Sklearn Parity Benchmarking
- Compared predictions against scikit-learn equivalents
- Achieved parity gap of only:

```text
0.014
