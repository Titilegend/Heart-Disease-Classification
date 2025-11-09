# Heart Disease Classification

**Predict Heart Disease Using Machine Learning**

This is an end-to-end machine learning pipeline capable of predicting whether a patient has heart disease from clinical parameters. Leveraging Python's data science ecosystem, this project walks through data analysis, model building, evaluation, and practical interpretation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Approach](#approach)
- [Algorithms & Metrics](#algorithms--metrics)
- [Results](#results)
- [Usage](#usage)
- [Requirements](#requirements)
- [References](#references)


---

## Project Overview

Heart disease is globally recognized as a leading health concern. Early and accurate predictions using accessible clinical data can have a significant impact. This notebook covers:

- Data loading and exploration
- Feature analysis and engineering
- Model building and comparison (Logistic Regression, KNN and Random Forest)
- Evaluation and visualization of results

---

## Data

- Dataset: [Cleveland Heart Disease Data (UCI)](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- Also available on [Kaggle](https://www.kaggle.com/datasets/sumaiyatasmeem/heart-disease-classification-dataset)
- Features include age, sex, chest pain type, cholesterol levels, fasting blood sugar, ECG results, and more.

---

## Approach

Step-by-step workflow:

1. **Problem Definition**: Predict heart disease status from given clinical indicators.
2. **Data Analysis**: Load, inspect, and visualize the dataset.
3. **Feature Engineering & Analysis**: Describe and transform variables as needed.
4. **Model Building**: Train and tune multiple classifiers.
5. **Evaluation**: Use statistical and graphical measures for model assessment.
6. **Experimentation & Interpretation**: Hyperparameter tuning, model comparison, and feature importance analysis.

---

## Algorithms & Metrics

### Algorithms Used
- **Logistic Regression**
- **K-Nearest Neighbors**
- **Random Forest**

### Model Evaluation Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**
- **ROC/AUC Curve**

---

## Results

- **Best Model:** Logistic Regression
- **Top Accuracy Achieved:** **88.5%**
- Other models (KNN, Random Forest) were evaluated, but Logistic Regression provided the highest performance and interpretability.
- Insights visualized via ROC curves, feature importances, and class frequency plots.

---

## Usage

**Getting Started:**
1. Clone this repository.
2. Ensure all required libraries are installed (see below).
3. Download `heart-disease.csv` and place in the project directory.
4. Open and run the notebook:
   ```
   end-to-end-heart-disease-classification.ipynb
   ```
5. Follow notebook cells sequentially for analyses, modeling, and results.

---

## Requirements

- Python 3.7+
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)

**Install all packages:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## References

- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- [Kaggle Dataset](https://www.kaggle.com/datasets/sumaiyatasmeem/heart-disease-classification-dataset)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Matplotlib Documentation](https://matplotlib.org/)
