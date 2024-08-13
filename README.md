# Project Title: Predicting Car Prices with Random Forest Regressor

## Table of Contents
1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Data Preparation](#data-preparation)
4. [Feature Engineering](#feature-engineering)
5. [Model Building](#model-building)
6. [Model Evaluation](#model-evaluation)
7. [Additional Models Tested](#additional-models-tested)
8. [Future Studies: Multicollinearity](#future-studies-multicollinearity)
9. [Model Saving & Deployment](#model-saving--deployment)
10. [Results and Conclusions](#results-and-conclusions)
11. [Next Steps](#next-steps)
12. [Getting Started](#getting-started)
13. [References](#references)

## Introduction
This project focuses on predicting car prices using machine learning techniques, with a primary focus on the Random Forest Regressor model. We also explored multiple linear regression and linear regression models for comparison. Feature engineering was performed to create new variables that could improve model performance. We have identified multicollinearity as a potential area for further study.

## Project Overview
The project is divided into several key sections:
- **Data Preparation:** Cleaning and preprocessing the dataset.
- **Feature Engineering:** Creating new features to enhance model performance.
- **Model Building:** Constructing and tuning the Random Forest Regressor, and testing other models for comparison.
- **Model Evaluation:** Evaluating model performance using metrics such as R2 Score, RMSE, and Cross-Validation.
- **Future Studies: Multicollinearity:** Planning a future study to explore the impact of multicollinearity on our model.
- **Model Saving & Deployment:** Saving the trained model for future use and possible deployment.

## Data Preparation
The dataset used for this project includes the following columns:
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

Steps taken:
- Handling missing values and outliers.
- Encoding categorical variables.
- Splitting the data into training and testing sets.

## Feature Engineering
To enhance the model's predictive capability, we engineered two new features:
- **`days_on_market`:** The number of days a car was listed for sale.
- **`car_age_at_sale`:** The age of the car at the time of sale.

These features were included to capture additional dimensions of car valuation that might influence the selling price.

## Model Building
### Random Forest Regressor
The primary model used in this project was the Random Forest Regressor, which was integrated into a pipeline for ease of use and reproducibility.

```python
rf3 = Pipeline([
    ('ColumnTransformer', transformer),  # Data transformation steps
    ('Model', RandomForestRegressor())  # Random Forest model
])
```

### Additional Models Tested
We also tested other models for comparison, including:
- **Multiple Linear Regression:** To understand the linear relationships between features and the target variable.
- **Linear Regression:** As a baseline model for comparison with more complex models.

```python
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
mlr_model = Pipeline([
    ('ColumnTransformer', transformer),
    ('Model', linear_model)
])

# Model training and evaluation for these models were performed to compare with Random Forest.
```

## Model Evaluation
We evaluated all models using various metrics:
- **R2 Score:** Indicates how well the model explains the variance in the target variable.
- **RMSE (Root Mean Squared Error):** Measures the average magnitude of the prediction errors.
- **Cross-Validation:** Provides insight into the model's performance on unseen data.

Final Evaluation Metrics for Random Forest Regressor:
- **Test R2:** 0.8551
- **Test RMSE:** 11448.49
- **Cross-Validated RMSE:** 12661.31
- **Cross-Validated R2:** 0.8081

## Future Studies: Multicollinearity
We have identified multicollinearity as a potential area for future exploration. This would involve:
- **Variance Inflation Factor (VIF):** To detect and measure the multicollinearity among features.
- **Feature Correlation Matrix:** To visualize correlations between features and identify any that might be highly correlated.

Conducting a multicollinearity study would help us understand the relationships between features better and ensure that the model is not adversely affected by redundant information.

## Model Saving & Deployment
The trained Random Forest Regressor model was saved using Python's `pickle` module. This allows for easy loading and deployment in future projects or production environments.

```python
import pickle

# Save the pipeline and model to a file
with open('rf3_pipeline.pkl', 'wb') as file:
    pickle.dump(rf3, file)

# Load the saved pipeline and model
with open('rf3_pipeline.pkl', 'rb') as file):
    loaded_pipeline = pickle.load(file)
```

## Results and Conclusions
The Random Forest Regressor model demonstrated strong predictive performance, making it a viable tool for estimating car prices based on the given features. The additional tests with multiple linear regression and linear regression models provided valuable insights into the linear relationships between features, while the future multicollinearity study is planned to further enhance model robustness.

## Next Steps
- **Multicollinearity Study:** Conduct the planned study to understand feature correlations and their impact on model performance.
- **Hyperparameter Tuning:** Further refine the model parameters to improve performance.
- **Feature Engineering:** Explore additional features that might enhance the model's accuracy.
- **Deployment:** Consider deploying the model in a production environment using platforms like AWS, Azure, or Heroku.
- **Model Interpretability:** Implement techniques like SHAP or LIME to better understand feature importance.

## Getting Started
To run this project locally:
1. Clone this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter notebook or Python scripts provided in the repository.
4. Use the saved model for predictions or further analysis.

## References
- [Scikit-learn Documentation](https://scikit-learn.org/stable/user_guide.html)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
- [LIME Documentation](https://github.com/marcotcr/lime)
