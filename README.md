# 💳 Credit Card Fraud Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Project-Complete-success)

---

## 📌 Project Overview

Credit card fraud detection is a classic **highly imbalanced classification problem**, where fraudulent transactions represent a very small fraction of the total data. This project focuses on building a **robust machine learning pipeline** that can accurately detect fraudulent transactions while handling class imbalance effectively.

The project uses **oversampling techniques** and **hyperparameter tuning** to improve model performance and generalization.

---

## 🎯 Objectives

* Detect fraudulent credit card transactions accurately
* Handle extreme class imbalance in the dataset
* Compare multiple machine learning models
* Improve performance using **oversampling** and **hyperparameter tuning**

---

## 📂 Dataset Description

* Dataset consists of anonymized credit card transactions
* Features are numerical (PCA-transformed for confidentiality)
* Target variable:

  * `0` → Legitimate Transaction
  * `1` → Fraudulent Transaction
* Highly imbalanced dataset (fraud cases < 1%)

---

## 🧠 Machine Learning Pipeline

### 1️⃣ Data Preprocessing

* Loaded dataset using **Pandas**
* Checked for missing values and duplicates
* Split data into features (`X`) and target (`y`)
* Performed **train-test split** to avoid data leakage

---

### 2️⃣ Handling Class Imbalance (Oversampling)

Fraud detection suffers from severe class imbalance, which can bias models toward predicting only legitimate transactions.

✔ **Oversampling Technique Used:** **SMOTE (Synthetic Minority Oversampling Technique)**

* Generates synthetic samples of the minority (fraud) class
* Helps models learn meaningful fraud patterns
* Applied **only on the training dataset** to avoid data leakage

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

---

### 3️⃣ Model Training

Multiple machine learning models were trained and evaluated, such as:

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)

Each model was trained on the **SMOTE-resampled training data**.

---

### 4️⃣ Hyperparameter Tuning

To improve performance, **hyperparameter tuning** was applied.

✔ **Technique Used:** **GridSearchCV**

* Performs exhaustive search over a parameter grid
* Uses cross-validation to avoid overfitting
* Selects best parameters based on evaluation metrics

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid.fit(X_train_resampled, y_train_resampled)
best_model = grid.best_estimator_
```

---

### 5️⃣ Model Evaluation

The models were evaluated on **unseen test data** using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC Score

Special emphasis was placed on **Recall**, as missing a fraud transaction is costlier than flagging a legitimate one.

---

## 📊 Results & Observations

* Oversampling significantly improved fraud detection
* Hyperparameter tuning boosted ROC-AUC score
* Random Forest / tuned models showed superior performance
* Balanced precision-recall tradeoff achieved

---

## 🛠️ Technologies Used

* **Python**
* **Pandas, NumPy**
* **Scikit-learn**
* **Imbalanced-learn (SMOTE)**
* **Matplotlib / Seaborn**

---

## ▶️ How to Run the Project

```bash
# Clone the repository
git clone https://github.com/CS-Prantik/Credit_Card_Fraud_Detection.git

# Navigate to project directory
cd Credit_Card_Fraud_Detection

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook
```

---

## 📈 Future Improvements

* Try advanced oversampling techniques (ADASYN, SMOTE-Tomek)
* Use ensemble methods like XGBoost or LightGBM
* Implement real-time fraud detection pipeline
* Add explainability using SHAP or LIME

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgements

* Scikit-learn Documentation
* UCI / Kaggle Credit Card Fraud Dataset

---

⭐ *If you find this project useful, feel free to star the repository!*
