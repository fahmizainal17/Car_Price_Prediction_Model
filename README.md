# **🚗 Predicting Car Prices with Random Forest Regressor**

---

## **📋 Table of Contents**
1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Data Preparation](#data-preparation)
4. [Feature Engineering](#feature-engineering)
5. [Model Building](#model-building)
6. [Model Evaluation](#model-evaluation)
7. [Additional Models Tested](#additional-models-tested)
8. [Model Saving & Deployment](#model-saving--deployment)
9. [Fundamentals](#fundamentals)
10. [Advanced Concepts](#advanced-concepts)
11. [Results and Conclusions](#results-and-conclusions)
12. [Next Steps](#next-steps)
13. [Getting Started](#getting-started)
14. [References](#references)

---

## **1. Introduction 🚀**
This project aims to predict car prices using machine learning techniques, focusing primarily on the Random Forest Regressor model. We also explore Multiple Linear Regression and Linear Regression models for comparison. Feature engineering was employed to create new variables that could enhance model performance and provide more accurate predictions.

## **2. Project Overview 🔍**
The project consists of the following key sections:
- **🧹 Data Preparation:** Cleaning and preprocessing the dataset.
- **🔧 Feature Engineering:** Creating new features to improve model accuracy.
- **🤖 Model Building:** Constructing and tuning the Random Forest Regressor, with comparison to other models.
- **📊 Model Evaluation:** Assessing model performance using metrics such as R² Score, RMSE, and Cross-Validation.
- **💾 Model Saving & Deployment:** Saving and deploying the trained model for future use.

---

## **🔧 Technologies Used**

<div>
    <h1 style="text-align: center;">Machine Learning with Python, Scikit-Learn, and Pickle</h1>
    <img style="text-align: left" src="https://img.icons8.com/color/48/000000/python.png" width="10%" alt="Python Logo" />
    <img style="text-align: left" src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" width="10%" alt="Scikit-Learn Logo" />
    <img style="text-align: left" src="https://raw.githubusercontent.com/FriendsOfPHP/pickle_logo/1961ac469151f43923eed29b2649ea26006e221a/pickle.svg" width="10%" alt="Pickle Logo" />
</div>
<br>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pickle](https://img.shields.io/badge/Pickle-77B063?style=for-the-badge&logo=python&logoColor=white)

---

## **3. Data Preparation 🛠️**
The dataset used includes various features that impact car pricing. The key steps taken during data preparation include:

- **📝 Columns:**
  - `car_brand`
  - `car_model`
  - `car_variant`
  - `car_year`
  - `car_engine`
  - `car_transmission`
  - `milage`
  - `accident`
  - `flood`
  - `color`
  - `purchase_date`
  - `sales_date`
  - `days_on_market` (Engineered feature)
  - `car_age_at_sale` (Engineered feature)
  - `price` (Target variable)

- **Key Steps:**
  - Handling missing values and outliers.
  - Encoding categorical variables.
  - Splitting the data into training and testing sets.

## **4. Feature Engineering 🛠️**
To boost the model's predictive capability, two new features were engineered:

- **`days_on_market`:** Represents the number of days a car was listed for sale.
- **`car_age_at_sale`:** Represents the car's age at the time of sale.

These features are designed to capture additional dimensions that might affect the car's selling price.

## **5. Model Building 🧠**
### **Random Forest Regressor 🌳**
The Random Forest Regressor is the primary model used, integrated into a pipeline for streamlined processing.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define the transformer for feature processing
transformer = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'
)

# Initialize and build the pipeline
rf_model = Pipeline([
    ('ColumnTransformer', transformer),
    ('Model', RandomForestRegressor())
])
```

## **6. Model Evaluation 📏**
Models were evaluated using the following metrics:

- **R² Score:** Indicates how well the model explains the variance in the target variable.
- **RMSE (Root Mean Squared Error):** Measures the average magnitude of prediction errors.
- **Cross-Validation:** Assesses model performance on unseen data.

**Final Evaluation Metrics for Random Forest Regressor:**
- **Test R²:** 0.8551
- **Test RMSE:** 11,448.49
- **Cross-Validated RMSE:** 12,661.31
- **Cross-Validated R²:** 0.8081

## **7. Additional Models Tested 🧪**
For comparative analysis, the following models were also tested:

- **Multiple Linear Regression:** Used to understand linear relationships between features and the target variable.
- **Linear Regression:** Served as a baseline model for comparison with more complex models.

```python
from sklearn.linear_model import LinearRegression

# Define the pipeline for Multiple Linear Regression
mlr_model = Pipeline([
    ('ColumnTransformer', transformer),
    ('Model', LinearRegression())
])
```

## **8. Model Saving & Deployment 💾**
The trained Random Forest Regressor model is saved using Python's `pickle` module, allowing for easy future use and deployment.

```python
import pickle

# Save the pipeline and model
with open('rf_model_pipeline.pkl', 'wb') as file:
    pickle.dump(rf_model, file)

# Load the saved model
with open('rf_model_pipeline.pkl', 'rb') as file:
    loaded_pipeline = pickle.load(file)
```

---

## **9. Fundamentals 🧩**
### **Understanding Regression Models:**
Regression models are used for predicting a continuous outcome variable based on one or more predictor variables. In this project, regression models like Random Forest and Linear Regression are utilized to predict car prices.

### **Feature Engineering:**
Feature engineering involves creating new input features from existing ones to improve model performance. In this project, engineered features like `days_on_market` and `car_age_at_sale` help capture additional information that may influence car prices.

### **Model Evaluation Metrics:**
- **R² Score:** A statistical measure that indicates how well the regression predictions approximate the real data points.
- **RMSE:** A metric that measures the average magnitude of the error, giving an idea of the prediction accuracy of the regression model.

---

## **10. Advanced Concepts 🚀**
### **Random Forest Regressor:**
A Random Forest Regressor is an ensemble learning method that builds multiple decision trees and merges their results to improve predictive accuracy and control overfitting. It’s robust against noise and provides better performance compared to individual decision trees.

### **Pipeline in Scikit-Learn:**
Pipelines are used in Scikit-Learn to automate the workflow of machine learning models, allowing for seamless data preprocessing and model training steps. In this project, the pipeline incorporates data transformation steps and the Random Forest model, ensuring streamlined and repeatable processes.

### **Model Deployment with Pickle:**
Pickle is a Python module used to serialize and deserialize Python objects. It allows saving the trained model, so it can be loaded and used for future predictions without retraining.

---

## **11. Results and Conclusions 🏁**
The Random Forest Regressor demonstrated strong performance, making it an effective tool for predicting car prices. The comparative analysis with multiple linear regression and linear regression models provided valuable insights into the relative performance of different approaches.

## **12. Next Steps ⏭️**
- **Deployment:** Explore deployment options using platforms such as AWS or Azure.
- **Model Interpretability:** Implement methods like SHAP or LIME to understand feature importance and make the model more interpretable.

## **13. Getting Started 🛠️**
To run this project locally:
1. **Clone this repository.**
2. **Install the required dependencies** using `pip install -r requirements.txt`.
3. **Execute the provided Jupyter notebook or Python scripts** to run the analysis.
4. **Use the saved model** for predictions or further analysis.

## **14. References 📚**
- [Scikit-learn Documentation](https://scikit-learn.org/stable/user_guide.html)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [SHAP Documentation](https://shap.readthed

ocs.io/en/latest/)
- [LIME Documentation](https://github.com/marcotcr/lime)

--- 
