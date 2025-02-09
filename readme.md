# Boston Airbnb Prediction of Listing Price

## Introduction

Airbnb has revolutionized the way people travel by providing a platform where hosts can offer accommodations ranging from shared rooms to entire properties. However, one of the most challenging aspects for hosts is determining the right price for their listings. Setting a competitive yet profitable price is influenced by various factors such as the property’s geographic location, property type, size, amenities. Hosts often struggle with pricing inconsistencies, which can lead to either underpricing, resulting in lost revenue, or overpricing, leading to reduced booking rates. This project aims to address these issues by predicting the price of Airbnb listings in Boston based on multiple features, offering hosts a data-driven approach to more accurately determine their pricing strategies.

## Problem statement

The challenge of predicting Airbnb listing prices lies in the multitude of factors influencing the price point. Geographic location, property type, amenities offered, and many other variables all play a crucial role in determining how much a property should cost. This variability creates difficulty for hosts in identifying the optimal price for their listings. Understanding the dynamics behind these factors can aid in reducing pricing errors and improve the booking rate and revenue for hosts.

​##Objectives

- Build a regression model to predict the price of Airbnb listings.​
- Identify significant features that impact the price of a listing.​
- Achieve a high level of accuracy in predicting listing prices to minimize underpricing or overpricing errors.​

## Dataset
The dataset used for this project was taken from the Boston Airbnb Open Data dataset on Kaggle. We have chosen to utilise listings.csv.  
- Source: https://www.kaggle.com/datasets/airbnb/boston?select=listings.csv


## Data Preprocessing
- Selection of relevant features for our problem statement and objective
- Conversion of datatypes and application of one-hot-encoding
- Imputation and removal of null values by deduction.
- New feature creation based on column with textual description, including applying a zero-shot classification BART NLP model to categorize textual descriptions of the size of accomodations into classes.


## Methodology
1. Data Cleaning and Preparation
3. Exploratory Data Analysis (EDA): Numerical distribution charts, class distribution charts, correlation analysis using boxplots, pearson's correlataion, point-biserial and kruskal-willis
4. Model Development: Multiple Linear Regression, Decision Tree Regression, Random Forest Regression, Support Vector Regression, Gradient Boosting Machine Regression, XGBoost Regression, Voting Regression Model, Neural Networks
5. Hyperparameter tuning: Grid Search CV
6. Model Evaluation (e.g. RMSE Learning curve, MAE, MSE, R²)

## Results

### Correlation Analysis
1. **Pearson’s Correlation (Numerical Variables)**:
   - **Strong Correlations**: Accommodates, bathrooms, bedrooms, beds, latitude, and longitude.
   - **Weak Correlations**: Availability_365, maximum_nights, minimum_nights.

2. **Point-Biserial Correlation (Boolean Variables)**:
   - **Strong Positive Correlations**: TV, air conditioning, cable TV, family/kid friendly, elevator, gym, doorman.
   - **Strong Negative Correlations**: Lock on the bedroom door, free parking on premises, allow smoking, cats, pets on the property.

3. **Kruskal-Wallis Test (Categorical Variables)**:
   - Significant differences observed across all categorical variables.

---

### Model Performance
| Model                               | MAE   | MSE   | R²    | Ranking |
|-------------------------------------|-------|-------|-------|---------|
| Multiple Linear Regression          | 0.273 | 0.138 | 0.681 | 4       |
| Decision Tree Regression            | 0.301 | 0.164 | 0.620 | 7       |
| Random Forest Regression            | 0.251 | 0.121 | 0.719 | 2       |
| Support Vector Regression           | 0.357 | 0.216 | 0.497 | 8       |
| Gradient Boosting Machine Regression| 0.258 | 0.123 | 0.714 | 3       |
| XGBoost Regression                  | 0.286 | 0.149 | 0.655 | 6       |
| Voting Regression Model             | 0.253 | 0.120 | 0.722 | 1       |
| Neural Network Model                | 0.270 | 0.141 | 0.673 | 5       |

- **Best Model**:  
  The Voting Regressor Model achieved the best overall performance with the lowest MAE (0.253), lowest MSE (0.120), and highest R² (0.722), indicating excellent accuracy and minimal error.  

- **Worst Model**:  
  The Support Vector Regression model performed the worst, with the highest MAE (0.357), highest MSE (0.216), and lowest R² (0.497), showing difficulty in capturing relationships effectively.



