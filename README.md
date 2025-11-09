# Heart Disease Classification

Predict Heart Disease Using Machine Learning

This project aims to develop an end-to-end machine learning pipeline to predict whether a patient has heart disease based on clinical parameters. It leverages Python's data science and machine learning libraries to analyze and model the data, striving for high accuracy and practical understanding.

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Approach](#approach)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling & Evaluation](#modeling--evaluation)
- [Usage](#usage)
- [Requirements](#requirements)
- [References](#references)

## Project Overview

Heart disease is a leading cause of death worldwide. Early and accurate prediction using clinical data can help save lives. The notebook walks through the entire process:
- Data loading and exploration
- Feature analysis
- Modeling (Logistic Regression, KNN, Random Forest, etc.)
- Evaluation and experimentation

## Data

- The dataset is sourced from the Cleveland data of the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease).
- You can also find a version on [Kaggle](https://www.kaggle.com/datasets/sumaiyatasmeem/heart-disease-classification-dataset).
- The dataset includes features like age, sex, chest pain type, cholesterol, fasting blood sugar, resting ECG results, and more.

## Approach

The notebook follows these steps:
1. **Problem Definition**: Given patient clinical parameters, predict heart disease presence.
2. **Data Analysis**: Loading, inspecting, and visualizing the dataset.
3. **Feature Analysis**: Understanding and preparing the variables.
4. **Model Building**: Training multiple machine learning models.
5. **Evaluation**: Assessing model performance and accuracy.
6. **Experimentation**: Hyperparameter tuning and comparison.

## Exploratory Data Analysis

- Analyze data distributions and relationships.
- Visualize feature impacts and heart disease frequency by sex and other factors.
- Check for missing values and outliers.

## Modeling & Evaluation

- Compare various models for accuracy, precision, recall, and F1-score.
- Final models are evaluated using metrics and ROC/AUC curves.
- A model with >95% accuracy will be considered successful for further development.

## Usage

**To run the analysis:**
1. Clone this repository.
2. Ensure you have the necessary libraries (see Requirements below).
3. Download the dataset (`heart-disease.csv`) and place it in the project directory.
4. Open and run the notebook:  
   ```
   end-to-end-heart-disease-classification.ipynb
   ```
5. Follow each notebook cell for step-by-step processing.

## Requirements

- Python 3.7+
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)

Install requirements (for local development):

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## References

- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- [Kaggle Dataset](https://www.kaggle.com/datasets/sumaiyatasmeem/heart-disease-classification-dataset)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Matplotlib Documentation](https://matplotlib.org/)
