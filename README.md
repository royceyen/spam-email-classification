# 📧 Spam Email Classification

## 📌 Overview

This project develops a machine learning pipeline to classify emails as **spam** or **non-spam**. The workflow includes data preprocessing, multiple classification algorithms, cost-sensitive learning, nested cross-validation, and comprehensive performance evaluation.

## 🎯 Problem Statement

Given the UCI Spambase dataset, the objectives are to:
1. Build and tune several classifiers to distinguish spam from non-spam emails.
2. Incorporate a **10∶1 cost ratio** to penalize false positives more heavily.
3. Use nested cross-validation to avoid overfitting during hyperparameter search.
4. Compare models on a range of metrics and choose the best overall and cost-sensitive classifiers.

## 📁 Dataset

- **Source**: UCI Machine Learning Repository – [Spambase](http://archive.ics.uci.edu/ml/datasets/Spambase)  
- **Files**:
  - `spambase.data`: 4,601 samples × 58 attributes  
  - `spambase.names` & `spambase.DOCUMENTATION`: attribute descriptions  

### Attributes

- **1–57**: continuous features (word/punctuation frequencies, capitalization patterns)  
- **58**: binary label (`1` = spam, `0` = non-spam)

## 🧠 Models Implemented

| Model                   | Key Hyperparameters            |
|-------------------------|--------------------------------|
| Logistic Regression     | `C`, `penalty`                 |
| k-NN Classifier         | `n_neighbors`                  |
| Decision Tree           | `max_depth`, `min_samples_leaf`|
| SVM                     | `C`, `kernel`, `gamma`         |
| MLP Neural Network      | `hidden_layer_sizes`, `alpha`  |
| Random Forest           | `n_estimators`, `max_depth`    |
| Gradient Boosting       | `n_estimators`, `learning_rate`|

- **Preprocessing**:  
  - Continuous features normalized via `StandardScaler`  
  - Binary indicators retained without scaling

- **Cost-Sensitive Learning**:  
  - Sample weights applied with a 10∶1 ratio (false positive cost = 10 × false negative)

- **Hyperparameter Tuning**:  
  - Nested cross-validation with `GridSearchCV` (inner loop) and `cross_val_score` (outer loop)

## 📊 Key Results

### 1. Best Models for Overall Accuracy (Default Settings)

| Model                  | CV Accuracy |
|------------------------|-------------|
| Gradient Boosting      | 0.9302      |
| Random Forest          | 0.9291      |
| MLPClassifier          | 0.9237      |

> **Top 3 (Default):** Gradient Boosting (0.9302), Random Forest (0.9291), MLPClassifier (0.9237)  
> *Among these, Gradient Boosting also achieved the highest detailed metrics:*  
> - Accuracy: **0.9566**  
> - Precision (spam class): **0.96**  
> - Recall (spam class): **0.93**  
> - F1-Score (spam class): **0.94**  
> - ROC AUC: **0.992**

---

### 2. Best Models for Cost-Sensitive Classification (Default Settings)

| Model             | Avg. Misclassification Cost |
|-------------------|-----------------------------|
| SVC               | 0.3586                      |
| Gradient Boosting | 0.3632                      |
| Random Forest     | 0.3873                      |

> **Top 3 (Default):** SVC (0.3586), Gradient Boosting (0.3632), Random Forest (0.3873)  
> *After hyperparameter tuning, Gradient Boosting achieved the lowest test‐set cost:* **0.1216** 

## 👤 Author

**Yi Hsiang (Royce) Yen**  
MS in Business Analytics, University of Minnesota  
