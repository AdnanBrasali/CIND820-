# Stock Market Trends Prediction with Machine Learning

Project Overview:

This project aims to predict stock market trends by leveraging machine learning techniques on a dataset spanning from 2013 to 2023, sourced from Kaggle and comprising approximately 25,000 data points. The primary objectives include forecasting future stock prices using Random Forest Regression, Decision Trees Regressor, and K-Nearest Neighbors Regressor, while also determining the directional movement of stock prices through Decision Trees Classifier, K-Nearest Neighbors Classifier, and Random Forest Classifier. The implementation is carried out in Python, with Scikit-learn for machine learning, Pandas for data handling, and Matplotlib and Seaborn for visualization. Evaluation metrics encompass Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), accuracy, precision, recall, and F1-score. The project integrates insights from a literature review, acknowledging the evolution of stock market prediction methodologies and identifying gaps in the literature. Analysis frameworks involve descriptive statistics, univariate, and bivariate analyses to uncover trends across ten major companies' stock prices. The ultimate goal is to offer actionable insights into stock market dynamics, contributing to the advancement of financial market analysis and aiding investors and analysts in making informed decisions.


Repository Contents

Plots: Directory containing visualizations generated during exploratory data analysis (EDA) and model evaluation stages.

EDA.ipynb: Jupyter notebook containing the exploratory data analysis of the stock market dataset. It includes data cleaning, visualization, and initial insights.

Full Analysis with and without adding new features.ipynb: Comprehensive Jupyter notebook showcasing the project's methodology, including data preprocessing, feature engineering, model training, and evaluation. The notebook compares model performances with and without newly added technical indicators.

README.md: This file, providing an overview of the project, its structure, and how to navigate the repository.

dataset.csv: The dataset used in the project, comprising daily stock market data for ten major companies over the last decade.

Stages of the Project

Literature Review: Examination of previous methodologies from fundamental analyses to advanced machine learning models for stock market prediction.

Data Collection: Compilation of a comprehensive dataset from Kaggle, including a decade’s worth of stock market data for ten major companies.

Data Preprocessing: Conversion of price columns to numeric formats, parsing of mixed date formats, and calculation of descriptive statistics.

Exploratory Data Analysis (EDA): Visualization of data distributions, detection and removal of outliers, and identification of key trends and correlations.

Feature Engineering: Creation of new features such as Momentum and Stochastic Oscillator indicators to enhance model prediction capabilities.

Modeling: Application of machine learning algorithms for regression (predicting stock prices) and classification (predicting the direction of stock price movements). This includes model training and evaluation.

Model Evaluation and Comparison: Assessment of model performances using metrics such as MAE, RMSE for regression, and accuracy, precision, recall, F1-score, and AUC for classification. Comparison of model outcomes with and without the addition of new features.

Feature Importance Analysis: Determination of the most influential features in predicting stock market trends.

Conclusion and Insights: Summary of findings, model performances, and recommendations for investors and market analysts.

Navigating the Repository

Begin by exploring the EDA.ipynb for initial data insights, followed by Full Analysis with and without adding new features.ipynb for a deep dive into the modeling process and outcomes. Visualizations in the Plots/ directory offer graphical representations of data and model evaluations, providing intuitive understanding of the analysis conducted.
