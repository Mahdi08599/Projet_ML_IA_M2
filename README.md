# 💳 Credit Card Fraud Detection: Research Reproduction

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/) 
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/sklearn-Machine_Learning-orange?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

## 📄 Project Overview
[cite_start]This project is a critical reproduction of the research paper [**"Enhancing credit card fraud detection using traditional and deep learning models"**](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1643292/full) (Albalawi & Dardouri, 2025). [cite: 1, 19, 26]

The goal was to replicate the study's findings regarding the superiority of **Random Forest** for fraud detection while improving the data engineering pipeline for better real-world robustness. The project includes a full **ML pipeline** (preprocessing, SMOTE, training) and a real-time **Streamlit web application**.

---

## 🎯 Key Objectives Achieved
- [cite_start]**Paper Reproduction:** Validated the authors' conclusion that Random Forest outperforms other traditional models[cite: 33].
- **Engineering Improvements:** Replaced the paper's *StandardScaler* with **RobustScaler** and limited tree depth to prevent the overfitting observed in the original study.
- **Real-Time Deployment:** Built an interactive **Streamlit app** capable of detecting fraud signatures in <50ms.
- [cite_start]**Class Imbalance Handling:** Implemented **SMOTE** (Synthetic Minority Over-sampling Technique) to handle the extreme 0.17% fraud ratio[cite: 32, 175].

---

## 📊 Model Performance: Us vs. The Paper

We compared our "Robust" Random Forest against the paper's reported metrics. While the paper achieved perfect Recall (likely due to overfitting), our model prioritizes **Precision** to minimize false alarms in a production setting.

| Metric | Paper's Random Forest | **Our Optimized Model** | Analysis |
| :--- | :--- | :--- | :--- |
| **F1-Score** |**0.8256**| **0.7847** | We matched the performance range while using safer hyperparameters. |
| **Precision** |**59%**| **74%** | **Superiority:** Our model generates significantly fewer false alarms. |
| **Recall** |**100.0%**| **84.0%** | **Realism:** We avoided overfitting (`max_depth=20`) for better generalization. |
| **ROC-AUC** | **0.9759** | **0.9796** | **Robustness:** Our model demonstrates slightly better ranking capability. |

---

## 🛠️ Project Workflow

### 1. Data Engineering
- [cite_start]**Log Transformation:** Applied `np.log1p` to the `Amount` feature to compress extreme outliers (up to $25k)[cite: 102].
- **Robust Scaling:** Used `RobustScaler` instead of Standard Scaling to handle the non-normal distribution of fraud data.
- [cite_start]**PCA Features:** Utilized anonymized features V1-V28 as described in the dataset[cite: 98].

### 2. Model Training
We benchmarked three algorithms on the **SMOTE-balanced** dataset:
1.  **Logistic Regression:** High recall (92%) but catastrophic precision (6%).
2.  **Decision Tree:** Balanced but unstable.
3.  **Random Forest (Champion):** Best trade-off between Precision and Recall.

### 3. Deployment (Streamlit)
The web app allows users to:
- Simulate "Normal" and "Fraud" profiles using real dataset samples.
- **Stress Test the Model:** Input high amounts ($1M) to verify the model ignores outliers if behavior (V1-V28) is normal.
- View real-time probability scores.

---

## 📂 Project Structure

```text
├── creditcard.csv         # Dataset (Git LFS or Local)
├── train_model.py         # Main ML pipeline (Preprocessing -> SMOTE -> Training)
├── app.py                 # Streamlit Dashboard code
├── requirements.txt       # Dependencies
└── README.md              # Project Documentation.

## 4. Model Selection & Training
- Implement and compare models:
  - **Traditional ML:** Logistic Regression, Decision Tree, Random Forest (Selected Champion)
- Train models on the SMOTE-balanced training set.

## 5. Model Fine-Tuning
- Optimize hyperparameters with **GridSearchCV** or **RandomizedSearchCV**.
- Adjust neural network architecture and learning rate.

## 6. Model Evaluation
- Evaluate models on the test set using:
  - Accuracy
  - Precision, Recall, F1-Score
  - ROC-AUC Score
- Compare performance of ML vs. DL models.

## 7. Deployment
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
- pandas
- scikit-learn
- Streamlit
- Git LFS (for large datasets)

---

