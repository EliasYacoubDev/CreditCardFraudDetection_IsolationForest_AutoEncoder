# Fraud Detection — Unsupervised vs Supervised Models

## Project Overview

This project aims to detect fraudulent transactions using two approaches:

1. **Unsupervised Learning**

   - **Autoencoder** — Learns to reconstruct "normal" transactions and flags anomalies based on reconstruction error.
   - **Isolation Forest** — Anomaly detection method that isolates outliers in feature space.
   - **Union Rule** — If **either** model predicts fraud, the transaction is flagged.
2. **Supervised Learning**

   - **XGBoost Classifier** — Trained directly on labeled fraud/non-fraud data for maximum accuracy.

The goal was to compare the performance of **unsupervised anomaly detection** vs **supervised classification** on the same dataset.

---

## Project Structure

├── data/ # Dataset files (not tracked in Git)
├── notebooks/
│ └── 02_models.ipynb # Main analysis & model training notebook
├── requirements.txt # Python dependencies
├── .gitignore # Ignore rules for Git
└── README.md # This file

---

## Workflow

### **Stage 1 — Unsupervised Models**

1. **Train Autoencoder**

   - Encodes transactions into a compressed representation.
   - Measures reconstruction error — high error indicates anomaly.
2. **Train Isolation Forest**

   - Randomly partitions feature space.
   - Outliers require fewer splits to isolate.
3. **Combine Predictions**

   - **Union Rule**: If either model predicts fraud → classify as fraud.
   - **Intersection Rule**: Only if both models agree → classify as fraud (lower recall).
4. **Best Unsupervised Choice**

   - **Union Rule** selected for high recall:
     - Recall: **0.9082**
     - ROC-AUC: **0.9298**

---

### **Stage 2 — Supervised Model**

1. **Train XGBoost Classifier**

   - Uses labeled data to learn patterns in fraudulent transactions.
   - Outperforms unsupervised methods when labels are available.
2. **Threshold Tuning**

   - Default threshold = 0.5
   - Also tested lower thresholds for higher recall.
3. **Best Supervised Results**

   - Recall: **0.8469** (default threshold)
   - Precision: **0.8830**
   - ROC-AUC: **0.9652**

---

## Key Results

| Model / Rule                              | Recall | Precision | F1-Score | ROC-AUC |
| ----------------------------------------- | ------ | --------- | -------- | ------- |
| **Autoencoder + Isolation (Union)** | 0.9082 | 0.0312    | 0.0604   | 0.9298  |
| **XGBoost (default)**               | 0.8469 | 0.8830    | 0.8646   | 0.9652  |

**Conclusion:**

- **Unsupervised (Union)** achieves very high recall, useful when missing a fraud case is very costly, but suffers from low precision (many false positives).
- **Supervised (XGBoost)** balances recall and precision much better when labeled training data is available.

---

## Learning Outcomes

- Learned how to implement **Autoencoder** and **Isolation Forest** for anomaly detection.
- Understood the trade-off between **recall** and **precision** in fraud detection.
- Implemented **threshold tuning** to prioritize recall in highly imbalanced datasets.
- Compared **unsupervised anomaly detection** vs **supervised classification** for the same problem.
- Learned that supervised models like **XGBoost** can significantly outperform unsupervised methods **if high-quality labeled data is available**.

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
Launch Jupyter Notebook
Open 02_models.ipynb and run the cells in order.

Requirements
See requirements.txt for the full list.
Main dependencies:

pandas

numpy

matplotlib

seaborn

scikit-learn

tensorflow

xgboost

jupyter

Notes
The dataset is highly imbalanced — less than 0.2% of transactions are fraudulent.

For real-world deployment, unsupervised methods are useful when labels are scarce.

If labels are available, supervised methods like XGBoost are preferred.

```
