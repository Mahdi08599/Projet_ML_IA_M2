# Credit Card Fraud Detection Project

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/) 
[![GitHub Repo](https://img.shields.io/badge/GitHub-Project-blue)](https://github.com/Mahdi08599/Projet_ML_IA_M2)

## Project Overview
This project aims to reproduce the findings of the paper [**"Enhancing credit card fraud detection using traditional and deep learning models"**](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1643292/full).  
The goal is to implement both **traditional machine learning** and **deep learning** models to detect fraudulent credit card transactions.  

This project also serves as a hands-on exercise to improve Python programming skills and understand the complete ML workflow: data preprocessing, model training, evaluation, and deployment.

---

## Project Objectives
- Reproduce the results from the original paper on a smaller scale.
- Implement **supervised ML models** (Logistic Regression, Random Forest, XGBoost) and **deep learning models** (Neural Networks).
- Build a **Streamlit application** to showcase predictions interactively.
- Follow best practices for data handling, model selection, and evaluation.

---

## Dataset
- **Source:** Credit card transactions dataset  
- **Format:** CSV  
- **Location:** [Google Drive Folder](https://drive.google.com/drive/folders/11DsuRJDWCwa1xyjogZwE6yjaKChsB3A2)  
- **Note:** If some data is unavailable, alternative sources or synthetic data are used.

---

## Project Workflow

### 1. Data Collection
- Download the dataset from the provided Google Drive link.
- Inspect the dataset for completeness and anomalies.

### 2. Data Exploration & Visualization
- Analyze transaction distributions, class imbalance, and feature correlations.
- Visualize patterns using **pandas**, **matplotlib**, and **seaborn**.

### 3. Data Preparation
- Handle missing values.
- Scale/normalize features if necessary.
- Split data into **training** and **testing** sets.
- Address class imbalance using techniques like **SMOTE**.

### 4. Model Selection & Training
- Implement models:
  - **Traditional ML:** Logistic Regression, Random Forest, XGBoost
  - **Deep Learning:** Feedforward Neural Network
- Train models on the training set.

### 5. Model Fine-Tuning
- Optimize hyperparameters with **GridSearchCV** or **RandomizedSearchCV**.
- Adjust neural network architecture and learning rate.

### 6. Model Evaluation
- Evaluate models on the test set using:
  - Accuracy
  - Precision, Recall, F1-Score
  - ROC-AUC Score
- Compare performance of ML vs. DL models.

### 7. Deployment
- Build a **Streamlit app**:
  - Accepts user input for transactions
  - Displays predictions
  - Shows model performance metrics

---

## Deliverables
1. **Presentation (.pptx)**:  
   - Explains the paper content  
   - Demonstrates how the study is reproduced (~20 slides)

2. **Python Package (.zip)**:  
   - **Back-end:** Data pipeline, model training, evaluation  
   - **Front-end:** Streamlit interactive application

---

## Tools & Libraries
- Python 3.10+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- TensorFlow / Keras
- Streamlit
- Git LFS (for large datasets)

---

