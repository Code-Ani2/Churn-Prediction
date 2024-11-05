# Churn Prediction with Logistic Regression

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data Description](#data-description)
- [Model Explanation](#model-explanation)
- [Results](#results)
- [Conclusion](#conclusion)


## Introduction

Customer churn prediction is a crucial aspect of business strategy, allowing companies to identify and retain customers who are likely to discontinue service. This project utilizes logistic regression, a statistical model that predicts binary outcomes, to analyze customer data and predict churn.

## Project Overview

In this project, we will build a logistic regression model to predict whether a customer will churn based on various features in the dataset. The code is implemented in Python and utilizes libraries such as Pandas, NumPy, and Scikit-learn.

### Objectives:

- Analyze customer behavior to identify factors contributing to churn.
- Build a predictive model using logistic regression.
- Evaluate the model's performance and accuracy.

## Installation

To run the churn prediction code, ensure you have the following dependencies installed. You can install them using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/yourusername/churn-prediction.git
    cd churn-prediction
    ```

2. Prepare your dataset in CSV format, ensuring it has the necessary features for prediction.

3. Open the `churn_prediction.py` file and modify the path to your dataset if necessary.

4. Run the script:
    ```bash
    python churn_prediction.py
    ```

5. The output will include model evaluation metrics and a visualization of the results.

## Data Description

The dataset consists of customer information and churn status. Here are the main features included:

- **CustomerID**: Unique identifier for each customer.
- **Age**: Age of the customer.
- **Gender**: Gender of the customer (male/female).
- **Tenure**: Duration of the customer relationship (in months).
- **Balance**: Account balance of the customer.
- **ProductsNumber**: Number of products the customer has.
- **IsActiveMember**: Indicates if the customer is an active member (1 or 0).
- **EstimatedSalary**: Estimated salary of the customer.
- **Exited**: Target variable (1 if the customer has churned, 0 otherwise).

Sure! Here’s a detailed description of the model explanation section, including the rationale for using logistic regression in your churn prediction project:

---

## Model Explanation

In this project, we employ **logistic regression** as our primary predictive modeling technique for customer churn analysis. Logistic regression is a statistical method used for binary classification problems, making it a suitable choice for predicting whether a customer will churn (exit) or remain with the service.

### Why Logistic Regression?

1. **Simplicity and Interpretability**:
   - Logistic regression is straightforward to implement and understand. The model outputs probabilities that can easily be interpreted as the likelihood of churn.
   - The coefficients of the model indicate the direction and strength of the relationship between each feature and the target variable (churn), allowing for clear insights into what factors are most influential.

2. **Binary Classification**:
   - Since our target variable (churn) is binary (1 for churn, 0 for no churn), logistic regression is particularly well-suited for this task. It models the probability that the target belongs to a particular category based on the input features.

3. **Efficiency**:
   - Logistic regression is computationally efficient, making it suitable for large datasets. It requires less computational power and time compared to more complex models like random forests or neural networks, which is advantageous when working with limited resources or time constraints.

4. **Handling Non-Linear Relationships**:
   - While logistic regression assumes a linear relationship between the log-odds of the outcome and the independent variables, it can be extended to capture non-linear relationships by using polynomial features or interaction terms.

5. **Robustness**:
   - Logistic regression is less sensitive to overfitting compared to more complex models, especially when the dataset is small relative to the number of features. This robustness is crucial in churn prediction, where we want to generalize well to unseen data.

### Model Development Steps

1. **Data Preprocessing**:
   - Data cleaning involved handling missing values, encoding categorical variables (e.g., gender), and scaling numerical features (e.g., balance, age) to ensure all variables are on a similar scale.

2. **Train-Test Split**:
   - The dataset was divided into training and testing subsets to evaluate the model's performance objectively. Typically, a split of 80% for training and 20% for testing is used.

3. **Fitting the Model**:
   - The logistic regression model was fitted to the training data, allowing it to learn the relationship between the features and the likelihood of customer churn.

4. **Model Evaluation**:
   - The model's performance was evaluated using metrics such as:
     - **Accuracy**: The proportion of true results (both true positives and true negatives) among the total number of cases examined.
     - **Precision**: The ratio of correctly predicted positive observations to the total predicted positives, indicating the quality of the positive class predictions.
     - **Recall (Sensitivity)**: The ratio of correctly predicted positive observations to all actual positives, highlighting the model’s ability to capture churners.
     - **F1 Score**: The harmonic mean of precision and recall, providing a single metric that balances both concerns.
     - **Confusion Matrix**: A matrix showing true positive, true negative, false positive, and false negative predictions, offering a comprehensive view of the model's performance.

By utilizing logistic regression, we are able to gain valuable insights into customer behavior and identify key factors influencing churn, ultimately aiding businesses in developing strategies for customer retention.

## Results

The results of the logistic regression model will be displayed as follows:

- Accuracy score of the model.
- Confusion matrix visualizing true vs. predicted values.
- Classification report detailing precision, recall, and F1-score.

## Conclusion

This project demonstrates how logistic regression can be employed to predict customer churn, providing insights into the factors influencing customer retention. The model's performance can be further improved by feature engineering, hyperparameter tuning, and using more complex algorithms.
