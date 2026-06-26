# Feature Selection using LASSO Regression

## Overview

This project demonstrates the implementation of **LASSO (Least Absolute Shrinkage and Selection Operator) Regression**, a powerful machine learning technique that combines **regularization** and **feature selection**.

LASSO applies **L1 regularization**, which shrinks the coefficients of less important features toward zero. Features whose coefficients become zero are automatically eliminated from the model, resulting in a simpler, more interpretable, and less overfitted machine learning model.

This project also explores how changing the **regularization parameter (alpha)** affects model complexity, feature selection, and predictive performance.

---

## Objectives

* Implement LASSO Regression using Scikit-learn.
* Perform automatic feature selection using L1 regularization.
* Evaluate model performance using the R² Score.
* Analyze feature coefficients after regularization.
* Experiment with different alpha values to understand their impact on model complexity and performance.

---

## Technologies Used

* Python
* Pandas
* Scikit-learn

---

## Project Structure

```text
LASSO-Feature-Selection/
│
├── lasso_regression.py
├── README.md
└── requirements.txt
```

---

## Dataset

The project uses a sample dataset to predict whether a student passes based on study habits and previous exam performance.

| Study Hours | Previous Exam Score | Pass |
| ----------- | ------------------- | ---- |
| 1           | 30                  | 0    |
| 2           | 40                  | 0    |
| 3           | 45                  | 0    |
| 4           | 50                  | 0    |
| 5           | 60                  | 0    |
| 6           | 65                  | 1    |
| 7           | 70                  | 1    |
| 8           | 75                  | 1    |
| 9           | 80                  | 1    |
| 10          | 85                  | 1    |

### Features

* **StudyHours**
* **PrevExamScore**

### Target Variable

* **Pass**

  * 0 = Fail
  * 1 = Pass

---

# LASSO Regression

## Description

LASSO (Least Absolute Shrinkage and Selection Operator) is a **linear regression algorithm with L1 regularization**.

Unlike standard linear regression, LASSO penalizes large coefficients by adding an L1 penalty term to the loss function. This causes less important feature coefficients to shrink toward zero, effectively removing them from the model.

This makes LASSO a powerful technique for:

* Feature Selection
* Regularization
* Reducing Overfitting
* Simplifying Machine Learning Models

---

## Project Workflow

### 1. Load Dataset

The dataset is loaded into a Pandas DataFrame and separated into:

* Features (X)
* Target Variable (y)

---

### 2. Split Dataset

The data is divided into:

* 80% Training Data
* 20% Testing Data

using `train_test_split()`.

---

### 3. Train LASSO Model

The model is initialized with a regularization parameter (**alpha**) and trained using the training dataset.

```python
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
```

---

### 4. Make Predictions

Predictions are generated on the testing dataset.

```python
y_pred = lasso_model.predict(X_test)
```

---

### 5. Evaluate Performance

The model performance is evaluated using the **R² Score**.

```python
r2 = r2_score(y_test, y_pred)
```

---

### 6. Analyze Feature Coefficients

The learned coefficients are displayed to identify important features.

```python
print(lasso_model.coef_)
```

Features with coefficients equal to **0** are automatically removed from the model.

Example Output:

```text
LASSO Coefficients:
[0.000, 0.022]
```

In this example:

* **StudyHours** → Removed from the model (coefficient = 0)
* **PrevExamScore** → Retained as an important predictor

---

## Model Workflow

```text
Load Dataset
      │
      ▼
Split Dataset
      │
      ▼
Initialize LASSO Model
      │
      ▼
Train Model
      │
      ▼
Predict Test Data
      │
      ▼
Evaluate Using R² Score
      │
      ▼
Analyze Feature Coefficients
      │
      ▼
Identify Important Features
```

---

## Model Evaluation

The model is evaluated using the **R² Score (Coefficient of Determination)**.

### R² Score

Measures how well the model explains the variation in the target variable.

```python
r2_score(y_test, y_pred)
```

### Interpretation

| R² Score  | Model Performance |
| --------- | ----------------- |
| 1.0       | Perfect Fit       |
| > 0.8     | Excellent         |
| 0.5 – 0.8 | Good              |
| < 0.5     | Poor              |

A higher R² Score indicates better predictive performance.

---

## Regularization Parameter (Alpha)

The **alpha** parameter controls the strength of regularization.

Example:

```python
for alpha in [0.01, 0.05, 0.1, 0.5, 1.0]:
    lasso_model = Lasso(alpha=alpha)
```

### Effect of Alpha

| Alpha Value  | Effect                                       |
| ------------ | -------------------------------------------- |
| Small Alpha  | Keeps more features, higher model complexity |
| Medium Alpha | Balanced feature selection                   |
| Large Alpha  | Removes more features, simpler model         |

Choosing the appropriate alpha helps balance model accuracy and complexity.

---

## Benefits of LASSO Regression

* Automatic Feature Selection
* Reduces Overfitting
* Simplifies Machine Learning Models
* Improves Interpretability
* Eliminates Redundant Features
* Works well with High-Dimensional Data
* Enhances Generalization on Unseen Data

---

## Advantages

* Performs feature selection automatically.
* Reduces model complexity.
* Prevents overfitting through regularization.
* Produces sparse and interpretable models.
* Effective when dealing with many input variables.

---

## Limitations

* Requires careful tuning of the alpha parameter.
* May eliminate correlated but useful features.
* Can underfit if regularization is too strong.

---

## Applications

LASSO Regression is widely used in:

* Feature Engineering
* Predictive Analytics
* Healthcare Diagnosis
* Financial Forecasting
* Customer Churn Prediction
* Marketing Analytics
* Bioinformatics
* Credit Risk Analysis
* Machine Learning Model Optimization

---

## Key Learning Outcomes

After completing this project, you will understand:

* LASSO Regression Fundamentals
* L1 Regularization
* Automatic Feature Selection
* Model Regularization
* R² Score Evaluation
* Feature Coefficient Analysis
* Hyperparameter Tuning (Alpha)
* Reducing Overfitting
* Building Interpretable Machine Learning Models

---

## Future Enhancements

* Apply LASSO to larger real-world datasets.
* Compare LASSO with:

  * Ridge Regression
  * Elastic Net
  * Backward Elimination
  * Forward Selection
  * Recursive Feature Elimination (RFE)
* Optimize alpha using GridSearchCV or Cross-Validation.
* Visualize coefficient shrinkage across different alpha values.
* Integrate LASSO into an end-to-end machine learning pipeline.

---

## Conclusion

This project demonstrates how **LASSO Regression** combines regularization and feature selection to build efficient machine learning models. By shrinking less important feature coefficients toward zero, LASSO automatically identifies the most relevant predictors, reducing model complexity while maintaining predictive performance. Experimenting with different alpha values provides valuable insight into balancing model simplicity and accuracy.

---

## Author

**Tejaswi**
Salesforce Developer | AI/ML Enthusiast | Aspiring Data Scientist

---

## License

This project is licensed under the MIT License.
