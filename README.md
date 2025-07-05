## 📊 Fraud Detection: Exploratory Data Analysis & Machine Learning

This project involves data exploration and machine learning modeling for fraud detection using a tabular dataset. The objective is to analyze the data, identify patterns and relationships, and build a model to predict fraudulent transactions.

### 📁 Files

* `code.ipynb` — Jupyter Notebook containing the entire workflow:

  * Data loading
  * Exploratory Data Analysis (EDA)
  * Preprocessing
  * Model training and evaluation

* `Fraud Detection Dataset.csv` — The dataset used for analysis (expected to be in the same directory).

---

### 🔍 Features

* **EDA** using pandas, matplotlib, seaborn to understand data distributions, correlations, and class imbalance.
* **Data preprocessing** including handling categorical variables and missing values.
* **Model training** using machine learning classifiers such as:

  * Random Forest
  * Logistic Regression
  * Support Vector Machine (SVM)
* **Model evaluation** using accuracy, F1-score, and confusion matrices.
* **SMOTE** used for balancing imbalanced datasets.

---

### 🛠️ Dependencies

Make sure you have the following Python packages installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

---

### 🚀 How to Run

1. Clone this repository or download the files.
2. Place `Fraud Detection Dataset.csv` in the same folder as the notebook.
3. Open and run `code.ipynb` in Jupyter Notebook or JupyterLab.

---

### 📈 Results

The notebook evaluates the performance of multiple models and compares them based on metrics like accuracy and F1-score. Class imbalance is addressed using SMOTE to improve prediction performance on minority classes.

---
 📌 Author

**Akash Roy**
MSc Statistics, IIT Kanpur


