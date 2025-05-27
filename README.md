# Diabetes Risk Prediction Using Machine Learning

## Overview

This project leverages supervised machine learning to predict the risk of diabetes using medical and lifestyle indicators. It was developed as a final project under the supervision of **Dr. Ali Zaart**.

Using Random Forest, along with ensemble and logistic models, this solution provides a robust and interpretable approach to identifying high-risk individuals, enabling early intervention and personalized healthcare strategies.

---

##  Problem Statement

A healthcare organization aims to reduce diabetes prevalence by identifying high-risk individuals based on their medical symptoms and demographic data. This allows for preventive actions and better health outcomes.

---

##  Dataset

- **Source**: [Kaggle - Diabetes Risk Prediction](https://www.kaggle.com/datasets/rcratos/diabetes-risk-prediction/data)
- **Features**:  
  - Age, Gender  
  - Polyuria, Polydipsia  
  - Sudden Weight Loss, Weakness, Polyphagia  
  - Genital Thrush, Visual Blurring, Itching  
  - Irritability, Delayed Healing, Partial Paresis  
  - Muscle Stiffness, Alopecia, Obesity  
- **Target**: `Class` (Positive / Negative diabetes diagnosis)

---

##  Data Processing Pipeline

###  Cleaning
- Removed duplicates
- Handled outliers using:
  - **Clipping** for numeric features
  - **Logistic Regression** for binary outliers

###  Transformation
- Label encoding of categorical features
- Train/Test split: 80% / 20%
- SMOTE applied for class imbalance

---

##  Exploratory Data Analysis

- **Correlation Heatmap** revealed strong predictors (e.g., Polyuria, Polydipsia)
- **Feature Importance** from Random Forest
- **Class Imbalance** addressed via SMOTE
- Visualized key insights using seaborn and matplotlib

---

##  Model Selection

| Model                | Accuracy | Notes |
|----------------------|----------|-------|
| Logistic Regression  | ~95%     | Baseline |
| Decision Tree        | ~96%     | High variance |
| **Random Forest**    | **97.5%**| Best performance |
| Voting Ensemble      | ~96.8%   | Stable alternative |

- Final model: **Random Forest**
- Justification: Handles mixed features, avoids overfitting, interpretable feature importance.

---

##  Evaluation

- **Confusion Matrix**
- **ROC Curve & AUC**:  
  - AUC: **0.97**  
  - Excellent discrimination between positive and negative cases

---

## 🧪 Cross-Validation

- 5-fold Cross-Validation Accuracy:  
  - Scores: `[0.96, 0.97, 0.94, 1.00, 1.00]`
  - **Mean**: `97.5%`  
  - **Std Dev**: `2.24%` — indicates strong generalization

---

##  Key Insights

- Most important features:
  - `Polyuria`, `Polydipsia`, `Age`, `Sudden Weight Loss`
- Majority of diabetics were aged **40–60**
- Positive class was more frequent → led to **initial overfitting**, resolved via **SMOTE**

---

##  Recommendations

- **Symptom-Based Screening**: Use polyuria and polydipsia for early alerts
- **Age-Focused Outreach**: Target 40–60 age range with routine checks
- **AI-Driven Triage Tools**: Integrate the model into healthcare platforms to assist clinicians

---

##  Project Structure

```bash
├── data/                     # Cleaned dataset
├── notebooks/               # Jupyter notebooks for EDA & modeling
├── src/                     # Scripts for preprocessing and training
├── models/                  # Saved model files
├── results/                 # Evaluation visuals: ROC, confusion matrix
├── README.md                # This file
```

## Tech Stack

- Language: Python
- Libraries: pandas, scikit-learn, seaborn, matplotlib, imblearn
- Model: RandomForestClassifier
- Tools: Jupyter Notebook, VS Code

## License

-This project is developed for academic purposes under the guidance of Dr. Ali Zaart. Please contact the author for reuse or collaboration.
