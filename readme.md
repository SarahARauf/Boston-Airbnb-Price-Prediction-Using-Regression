# Boston Airbnb Prediction of Listing Price

## Introduction

Airbnb has revolutionized the way people travel by providing a platform where hosts can offer accommodations ranging from shared rooms to entire properties. However, one of the most challenging aspects for hosts is determining the right price for their listings. Setting a competitive yet profitable price is influenced by various factors such as the property’s geographic location, property type, size, amenities. Hosts often struggle with pricing inconsistencies, which can lead to either underpricing, resulting in lost revenue, or overpricing, leading to reduced booking rates. This project aims to address these issues by predicting the price of Airbnb listings in Boston based on multiple features, offering hosts a data-driven approach to more accurately determine their pricing strategies.

## Problem statement

The challenge of predicting Airbnb listing prices lies in the multitude of factors influencing the price point. Geographic location, property type, amenities offered, and many other variables all play a crucial role in determining how much a property should cost. This variability creates difficulty for hosts in identifying the optimal price for their listings. Understanding the dynamics behind these factors can aid in reducing pricing errors and improve the booking rate and revenue for hosts.

​## Objectives
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
3. Exploratory Data Analysis (EDA): Normal distribution charts on numeric continuous variables,  
4. Model Development (e.g., Linear Regression, Random Forest, Neural Networks)
5. Model Evaluation (e.g., Accuracy, F1 Score)


