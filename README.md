# ⚠️ Dangerous Activity Classification | Neural Networks & Scikit-learn

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/mostafaelsayed7/animal-condition-classification)

> MLP-based binary classifier achieving **99.3% accuracy** on structured behavioral data —
> minimizing false negatives in safety-critical environments.

---

## 📌 Project Overview

Built a **Multi-Layer Perceptron (MLP)** binary classifier using Scikit-learn to predict
whether a given behavioral condition is dangerous or not, trained on 871 structured records.
The pipeline includes full preprocessing — cleaning, one-hot encoding, and train-test splitting.
Directly applicable in:

- 🏭 Workplace safety monitoring systems
- 📹 Smart surveillance & anomaly detection
- 🐾 Animal behavior risk assessment
- 🏥 Patient condition monitoring in healthcare

---

## 🔧 Full Pipeline

```
Raw CSV Data (871 records)
    ↓
Data Cleaning — drop null values
    ↓
One-Hot Encoding — pd.get_dummies (drop_first=True)
    ↓
Train-Test Split — 67% train / 33% test (random_state=42)
    ↓
MLPClassifier (hidden_layer_sizes=(100,), max_iter=1000)
    ↓
Prediction & Accuracy Scoring
    ↓
99.3% Accuracy — minimized false negatives
```

---

## 📊 Results

| Metric | Value |
|---|---|
| Dataset | Animal Disease Dataset (871 records) |
| Test Accuracy | **99.3%** |
| Model | MLPClassifier |
| Hidden Layers | 1 layer × 100 neurons |
| Max Iterations | 1,000 |
| Test Size | 33% |
| Random State | 42 |

---

## 🔧 Preprocessing Pipeline

| Step | Detail |
|---|---|
| Null Handling | Dropped rows with missing values |
| Encoding | One-hot encoding via `pd.get_dummies(drop_first=True)` |
| Target Variable | `Dangerous_Yes` (binary) |
| Train-Test Split | 67% / 33% stratified split |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `scikit-learn` | MLPClassifier, train-test split, accuracy scoring |
| `pandas` | Data loading, cleaning, one-hot encoding |
| `numpy` | Array operations |
| `seaborn` / `matplotlib` | Data visualization & EDA |
| `kagglehub` | Dataset download from Kaggle |

---

## ▶️ How to Run

1. View on Kaggle (recommended — dataset is pre-loaded):

[![Kaggle](https://img.shields.io/badge/View%20on-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/mostafaelsayed7/animal-condition-classification)

2. Or install dependencies locally:
```bash
pip install scikit-learn pandas numpy seaborn matplotlib kagglehub
```

3. Download the dataset:
```python
import kagglehub
path = kagglehub.dataset_download('gracehephzibahm/animal-disease')
```

4. Update the CSV path and run all cells.

---

## 📁 Project Structure

```
dangerous-activity-classification/
│
├── notebook.ipynb     # Full MLP pipeline
└── README.md          # Project documentation
```

---

## 👤 Author

**Mostafa El-Sayed** — Data Scientist | Computer Vision Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/mostafa-nfc)
[![Gmail](https://img.shields.io/badge/Gmail-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:m.e.2172000@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/MostafaELsayed-217)
