# 📧 Email Spam Detection

Spam detection using Logistic Regression built from scratch with NumPy.

## 📊 Dataset
- SMS Spam Collection — 5,572 messages
- Ham: 4,825 | Spam: 747

## ⚙️ Pipeline
1. Text cleaning with Regex
2. TF-IDF Vectorization (bigram, 3000 features)
3. Logistic Regression from scratch (Gradient Descent + class weights)

## 📈 Results
| Metric    | Ham  | Spam |
|-----------|------|------|
| Precision | 0.98 | 0.86 |
| Recall    | 0.98 | 0.89 |
| F1-score  | 0.98 | 0.87 |
| Accuracy  | 96.59%      |

## 🔧 How it works
- Sigmoid function for probability estimation
- Gradient Descent for weight optimization
- Class weights to handle imbalanced data (86% ham / 14% spam)

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 📓 Run

```bash
jupyter notebook spamdetection.ipynb
```

## 📊 Dataset

https://archive.ics.uci.edu/dataset/143/sms+spam+collection
