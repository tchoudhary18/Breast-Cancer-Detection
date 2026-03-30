# 🎗️ Breast Cancer Classification using Machine Learning

A complete end-to-end Machine Learning project that classifies breast cancer tumors as **Malignant** or **Benign** using the Wisconsin Breast Cancer Dataset. Five different classification algorithms are trained, evaluated, and compared to find the best-performing model.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Workflow](#project-workflow)
- [Models Used](#models-used)
- [Results](#results)
- [Visualizations](#visualizations)
- [Project Structure](#project-structure)
- [Key Concepts](#key-concepts)

---

## 📖 Project Overview

Breast cancer is one of the most common cancers worldwide. Early and accurate detection significantly improves patient outcomes. This project builds a machine learning pipeline that:

- Performs **Exploratory Data Analysis (EDA)** to understand the data
- Applies **preprocessing** (scaling, PCA, outlier removal)
- Trains and compares **5 classification models**
- Evaluates performance using accuracy, confusion matrix, ROC-AUC, and classification reports
- Saves the best model to disk using **Pickle** for future use

---

## 📊 Dataset

- **Source:** [Wisconsin Breast Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) (built into scikit-learn)
- **Samples:** 569 patient records
- **Features:** 30 numerical features (mean, standard error, and worst values of cell nucleus measurements)
- **Target:** Binary — `0` = Malignant, `1` = Benign
- **Class Distribution:** ~62% Benign, ~38% Malignant

### Feature Groups

| Group | Columns | Description |
|---|---|---|
| `_mean` | 0–9 | Mean values of cell measurements |
| `_se` | 10–19 | Standard error of measurements |
| `_worst` | 20–29 | Worst (largest) values of measurements |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `scikit-learn` | ML models, preprocessing, evaluation |
| `pandas` | Data loading, manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting and visualizations |
| `seaborn` | Statistical visualizations |
| `pickle` | Model serialization |

---

## 🔄 Project Workflow

```
Raw Dataset (sklearn)
        │
        ▼
   Load & Save CSV
        │
        ▼
Exploratory Data Analysis
(histograms, heatmap, box plots, pair grid)
        │
        ▼
  Preprocessing
(StandardScaler, LabelEncoder, Outlier Removal)
        │
        ▼
Dimensionality Reduction (PCA: 30 → 10 components)
        │
        ▼
  Train / Test Split (70% / 30%, stratified)
        │
        ▼
  Train 5 ML Models
        │
        ▼
  Evaluate & Compare
(Accuracy, Confusion Matrix, ROC-AUC, Cross-Validation)
        │
        ▼
  Save Best Model (Pickle)
```

---

## 🤖 Models Used

| Model | Key Idea |
|---|---|
| **Support Vector Machine (SVM)** | Finds the optimal hyperplane separating classes; tested with linear, RBF, and polynomial kernels |
| **Logistic Regression** | Outputs class probability; tuned with GridSearchCV |
| **Gaussian Naive Bayes** | Probabilistic classifier based on Bayes' theorem |
| **Random Forest** | Ensemble of 100 decision trees, majority vote |
| **K-Nearest Neighbors (KNN)** | Classifies based on k=26 nearest neighbors; optimal k found by error-rate plot |

---

## 📈 Results

| Model | Test Accuracy |
|---|---|
| ✅ Support Vector Machine | **97.08%** |
| Logistic Regression | 95.32% |
| K-Nearest Neighbors | 95.32% |
| Random Forest | 92.98% |
| Naive Bayes | 91.81% |

> **SVM achieved the highest accuracy at 97.08%**, making it the best model for this dataset.

Additional evaluation metrics used:
- **Confusion Matrix** — breakdown of TP, TN, FP, FN
- **Classification Report** — Precision, Recall, F1-score per class
- **ROC Curve & AUC Score** — model discrimination ability
- **3-Fold & 5-Fold Cross-Validation** — generalization performance

---

## 📉 Visualizations

The project includes the following visualizations:

- 📊 **Histograms** — distribution of each feature group (mean, SE, worst)
- 🌊 **KDE / Density Plots** — smoothed distribution curves
- 🔥 **Correlation Heatmap** — feature correlation matrix
- 📦 **Box Plots** — outlier detection per feature group
- 🔵 **Pair Grid** — pairwise feature relationships colored by class
- 🌀 **PCA Scatter Plot** — 2D visualization of class separability
- 🧮 **Confusion Matrix** — visual grid of prediction results
- 📉 **ROC Curve** — true positive vs false positive rate
- 📏 **KNN Error Rate Plot** — optimal k selection
- 📊 **Model Accuracy Bar Chart** — final comparison of all models
- 🗺️ **SVM Decision Boundary** — visualization for linear, RBF, and polynomial kernels

---


### Output Files

After running, the following files will be generated:
- `cancer.csv` — the dataset saved as a CSV file
- `breastcancer_model_saved.pkl` — the trained model saved with Pickle

### Load the Saved Model

```python
import pickle

model = pickle.load(open('breastcancer_model_saved.pkl', 'rb'))
prediction = model.predict(new_data)
```

---

## 📁 Project Structure

```
breast-cancer-classification/
│
├── breast_cancer_classification.ipynb   # Main Jupyter Notebook
├── breast_cancer_classification.py      # Python script version
├── cancer.csv                           # Generated dataset (auto-created)
├── breastcancer_model_saved.pkl         # Saved trained model (auto-created)
└── README.md                            # This file
```

---

## 💡 Key Concepts

- **Supervised Learning** — Learning from labelled data (we know the correct answer for training samples)
- **Binary Classification** — Predicting one of two classes (malignant vs benign)
- **StandardScaler** — Normalizes features to zero mean and unit variance; essential for SVM and KNN
- **PCA** — Reduces 30 features to 10 principal components while retaining maximum variance
- **Cross-Validation** — Splits data into multiple folds for a more robust accuracy estimate
- **GridSearchCV** — Exhaustive search over hyperparameter combinations to find the best settings
- **ROC-AUC** — Measures how well the model separates classes across all decision thresholds
- **Pickle** — Serializes Python objects (like trained models) to disk for later use

---

*If you found this project helpful, please consider giving it a ⭐ on GitHub!*
