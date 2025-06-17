# 📧 Spam Email Detection

## 📁 Project Overview

This repository implements a comprehensive pipeline for **classifying emails** as spam or non-spam. Key features:

- Data preprocessing and feature scaling
- Multiple classification algorithms
- Cost-sensitive learning (10:1 false positive cost)
- Nested cross-validation for unbiased performance
- Detailed evaluation metrics and visualizations

---

## 📂 Repository Structure

```
spam-email-detection/
├── data/
│   └── spambase.data
├── notebooks/
│   └── Spam_Email_Prediction.ipynb
├── src/
│   └── data_preprocessing.py
│   └── model_training.py
│   └── evaluation.py
├── requirements.txt
└── README.md
```

---

## 🔧 Installation & Usage

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/spam-email-detection.git
   cd spam-email-detection
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**  
   ```bash
   jupyter notebook notebooks/Spam_Email_Prediction.ipynb
   ```

Or execute the scripts in `src/` for modular workflows.

---

## 🛠 Methodology

1. **Data Preprocessing**  
   - Load dataset (`spambase.data`)
   - Normalize continuous features with `StandardScaler`
   - Retain binary indicators without scaling

2. **Modeling & Tuning**  
   - Algorithms: Logistic Regression, k-NN, Decision Tree, SVM, MLP Neural Network, Random Forest, Gradient Boosting  
   - Hyperparameter tuning via nested `GridSearchCV`

3. **Cost-Sensitive Learning**  
   - Apply a **10:1 cost ratio** for false positives vs. false negatives using sample weights

4. **Evaluation & Visualization**  
   - Metrics: Accuracy, Precision, Recall, F1-Score, ROC AUC, Average Misclassification Cost  
   - Visuals: Combined ROC curves, CONFusion matrices, lift charts  

---

## 📊 Key Results

| Model                   | ROC AUC | Avg. Misclassification Cost |
|-------------------------|---------|-----------------------------|
| Logistic Regression     | 0.95    | 1.8                         |
| Random Forest           | 0.97    | 1.3                         |
| Gradient Boosting       | 0.98    | 1.1                         |
| MLP Neural Network      | 0.96    | 1.6                         |

*Gradient Boosting with cost weighting achieved the best performance.*

---

## 👤 Author

**Yi Hsiang (Royce) Yen**  
Master of Science in Business Analytics  
University of Minnesota  
