# Credit Card Fraud Detection
This project detects fraudulent transactions using machine learning.

## Overview
- *Goal:* Classify transactions as fraudulent or legitimate.
- *Dataset:* Kaggle Credit Card Fraud Detection
- *Tech Stack:* Python, Pandas, Scikit-learn, Jupyter Notebook

## Dataset Summary
> Large dataset not included (over 100MB).  
> Download from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions
- 492 fraud cases
- Features: `V1` to `V28`, `Amount`, `Time`, `Class`

## ML Pipeline
- Preprocessing and normalization
- Anomaly detection & classification models
- Performance evaluation using accuracy, precision, recall, F1-score
  
## How to Run
1. Download the dataset from Kaggle
2. Place it in the same folder
3. Run the notebook: `Fraud Detection code.ipynb`

## 🙋 Author

**Neha Mariyala**  
GitHub: [@MariyalaNeha](https://github.com/MariyalaNeha)

# Credit Card Fraud Detection 💳🔍

This project is aimed at detecting fraudulent credit card transactions using machine learning techniques. It utilizes a real-world dataset and applies multiple models to classify transactions as fraudulent or genuine.

## 📂 Project Structure

- `credit_card_sample.csv` – Sample of the original dataset (20%)
- `Fraud Detection code.ipynb` – Jupyter notebook with full workflow
- `.gitignore` – Ignores large original file (`credit card.csv`)

## 📊 Dataset Overview

The dataset contains 284,807 credit card transactions with 30 anonymized features (`V1`–`V28`), `Amount`, and `Class` (1 = Fraud, 0 = Legit).

- **Class distribution** is highly imbalanced:
  - Genuine: 284,315
  - Fraud: 492

## 🧪 Process Overview

1. **Data Sampling & Cleaning**
   - Used 20% sample to reduce size
   - Removed `Time` column
   - Scaled `Amount` using `StandardScaler`

2. **EDA & Visualization**
   - Class imbalance visualized using `seaborn` countplot

3. **Modeling Techniques**
   - **Logistic Regression**
     - ROC AUC: **0.95**
     - F1-Score (Fraud): 0.72
   - **Random Forest Classifier**
     - ROC AUC: **0.99**
     - F1-Score (Fraud): 0.89
     - Best overall performer
   - **Isolation Forest** (Anomaly Detection)
     - Lower recall (0.20), but useful for unsupervised scenarios

## 📈 Results Summary

| Model               | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) | ROC AUC |
|--------------------|-------------------|----------------|------------------|---------|
| Logistic Regression| 0.83              | 0.64           | 0.72             | 0.96    |
| Random Forest       | 0.94              | 0.85           | 0.89             | 0.999   |
| Isolation Forest    | 0.33              | 0.20           | 0.25             | —       |

## 📌 Technologies Used

- Python (Pandas, NumPy, Seaborn, Matplotlib)
- Scikit-learn: LogisticRegression, RandomForest, IsolationForest
- Jupyter Notebook

## ✅ Conclusion

- Random Forest performed best for fraud detection in this dataset.
- Class imbalance requires special handling (e.g., sampling, `class_weight`)
- Unsupervised models like Isolation Forest may help when labels are unavailable.

## 🚀 Future Improvements

- Try SMOTE or other resampling techniques
- Hyperparameter tuning for Random Forest
- Use LSTM or Autoencoders for time-based anomaly detection

---


