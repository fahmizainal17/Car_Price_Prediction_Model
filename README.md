# Project Title: Predicting Car Prices with Random Forest Regressor

## Table of Contents
1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Data Preparation](#data-preparation)
4. [Feature Engineering](#feature-engineering)
5. [Model Building](#model-building)
6. [Model Evaluation](#model-evaluation)
7. [Additional Models Tested](#additional-models-tested)
8. [Model Saving & Deployment](#model-saving--deployment)
9. [Results and Conclusions](#results-and-conclusions)
10. [Next Steps](#next-steps)
11. [Getting Started](#getting-started)
12. [References](#references)

## Introduction
This project aims to predict car prices using machine learning techniques, with a primary focus on the Random Forest Regressor model. The project includes exploring multiple linear regression and linear regression models for comparative purposes. Feature engineering was utilized to create new variables that could potentially enhance model performance.

## Project Overview
The project comprises the following key sections:
- **Data Preparation:** Involves cleaning and preprocessing the dataset.
- **Feature Engineering:** Includes creating new features to improve model accuracy.
- **Model Building:** Focuses on constructing and tuning the Random Forest Regressor, with comparison to other models.
- **Model Evaluation:** Evaluates model performance using metrics such as R2 Score, RMSE, and Cross-Validation.
- **Model Saving & Deployment:** Details on saving and deploying the trained model for future use.

## Data Preparation
The dataset used in this project includes the following columns:
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

Key steps taken:
- Handling missing values and outliers.
- Encoding categorical variables.
- Splitting the data into training and testing sets.

## Feature Engineering
To enhance the model's predictive capability, two new features were engineered:
- **`days_on_market`:** Represents the number of days a car was listed for sale.
- **`car_age_at_sale`:** Represents the car's age at the time of sale.

These features are intended to capture additional dimensions of car valuation that might affect the selling price.

## Model Building
### Random Forest Regressor
The primary model used is the Random Forest Regressor, integrated into a pipeline for streamlined processing.

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

### Additional Models Tested
For comparative analysis, the following models were also tested:
- **Multiple Linear Regression:** Assessed to understand linear relationships between features and the target variable.
- **Linear Regression:** Served as a baseline model for comparison with more complex models.

```python
from sklearn.linear_model import LinearRegression

# Define the pipeline for Multiple Linear Regression
mlr_model = Pipeline([
    ('ColumnTransformer', transformer),
    ('Model', LinearRegression())
])
```

## Model Evaluation
Models were evaluated using the following metrics:
- **R2 Score:** Indicates how well the model explains the variance in the target variable.
- **RMSE (Root Mean Squared Error):** Measures the average magnitude of prediction errors.
- **Cross-Validation:** Assesses model performance on unseen data.

Final Evaluation Metrics for Random Forest Regressor:
- **Test R2:** 0.8551
- **Test RMSE:** 11448.49
- **Cross-Validated RMSE:** 12661.31
- **Cross-Validated R2:** 0.8081

## Model Saving & Deployment
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

## Results and Conclusions
The Random Forest Regressor demonstrated strong performance, making it an effective tool for predicting car prices. The comparative analysis with multiple linear regression and linear regression models provided insights into their relative performance.

## Next Steps
- **Deployment:** Explore deployment options using platforms such as AWS or Azure
- **Model Interpretability:** Implement methods like SHAP or LIME to understand feature importance.

## Getting Started
To run this project locally:
1. Clone this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Execute the provided Jupyter notebook or Python scripts.
4. Use the saved model for predictions or further analysis.

## References
- [Scikit-learn Documentation](https://scikit-learn.org/stable/user_guide.html)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
- [LIME Documentation](https://github.com/marcotcr/lime)
