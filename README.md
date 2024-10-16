
# Data Engineering for Motor Insurance Campaign

This repository contains the capstone project for the course COMP6670 at the University of Kent, supervised by Dr. Jian Zhang. The project aims to transform raw motor insurance data into actionable insights by leveraging data engineering techniques and machine learning algorithms to predict whether a customer contacted during a motor insurance campaign will purchase car insurance.

## Table of Contents
1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Project Structure](#project-structure)
4. [Data Description](#data-description)
5. [Methodology](#methodology)
6. [Machine Learning Models](#machine-learning-models)
7. [Results](#results)
8. [Getting Started](#getting-started)
9. [Future Work](#future-work)
10. [References](#references)

## Abstract
The project focuses on the context of a bank offering car insurance services, utilizing historical and current campaign data to identify potential clients for marketing insurance products. Machine learning algorithms are employed to forecast whether individuals will purchase car insurance based on features such as demographics, past interactions, and financial details. The final deliverable includes a predictive Python package named `insurance-predictor` for use in future campaigns.

## Introduction
In today's data-driven world, using data analytics to enhance customer outreach and marketing efficiency is essential. This project uses machine learning techniques to analyze motor insurance campaign data, aiming to improve customer targeting strategies and predict campaign outcomes. The implemented models include:
- **Random Forest**
- **XGBoost**
- **Neural Networks**

## Project Structure
The repository is organized as follows:
- **[Capstone_final.ipynb](Capstone_final.ipynb)**: Main code for data preprocessing, model training, and evaluation.
- **[carInsurance_train.csv](carInsurance_train.csv)**: Training dataset.
- **[carInsurance_test.csv](carInsurance_test.csv)**: Test dataset.
- **[insuarnce_predictor_demo.ipynb](insuarnce_predictor_demo.ipynb)**: Demonstrates the usage of the `insurance-predictor` package.
- **[model_prediction_test_data.csv](model_prediction_test_data.csv)**: Model predictions on the test data.
- **Plots and Graphs**:
  - [Feature Importance XGB plot](Feature%20Importance%20XGB%20plot.png)
  - [ROC curve for optimized neural network](ROC%20curve%20for%20optimized%20neural%20network.png)
  - [Training and Validation Loss for neural network](Training%20and%20Validation%20Loss%20for%20neural%20network.png)
- **[insurance_predictor/](insurance_predictor/)**: Files used to build the `insurance-predictor` Python package.
- **[index.html](index.html)**: Submission index for project files.

## Data Description
The dataset consists of customer information and campaign data, including demographic details, communication methods, financial status, and historical campaign outcomes. Key features:
- **Age, Job, Education, Balance**: Demographic and financial attributes.
- **Communication**: Type of contact (cellular, telephone, unknown).
- **Campaign Interaction**: Details like `CallDuration`, `LastContactMonth`, and `Outcome`.
- **Target Variable**: `CarInsurance`, indicating whether the customer purchased car insurance (1 for yes, 0 for no).

Data preprocessing steps included handling missing values, encoding categorical variables, and feature engineering.

## Methodology
The project's methodology followed a classical machine learning approach:
1. **Exploratory Data Analysis (EDA)**: Uncovered patterns and data characteristics.
2. **Data Preprocessing**: Addressed missing values, scaled features, and performed encoding.
3. **Feature Engineering**: Created new features such as `CallDuration`.
4. **Model Training and Evaluation**: Split data into training and validation sets (80/20), trained models, and assessed performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

## Machine Learning Models
### 1. Random Forest
   - **Purpose**: Leveraged decision trees for classification.
   - **Outcome**: Achieved 85% accuracy with a strong ability to predict negative cases.

### 2. XGBoost
   - **Purpose**: Utilized gradient boosting for higher accuracy.
   - **Outcome**: Provided high accuracy and was the most effective model with an AUC of 0.92.

### 3. Neural Networks
   - **Purpose**: Captured complex non-linear relationships in the data.
   - **Outcome**: Showed good discriminative power with an AUC of 0.91, though some overfitting was observed.

### Hyperparameter Optimization
- **XGBoost**: Bayesian optimization improved model parameters.
- **Neural Networks**: `keras-tuner` was used to optimize architecture and hyperparameters.

## Results
### Evaluation Metrics
- **Accuracy, Precision, Recall, F1-Score**: All models performed well, with XGBoost and Random Forest showing the highest accuracies.
- **ROC-AUC**: Demonstrated that all models were effective in discriminating between customers who would or wouldn't purchase car insurance.
- **Feature Importance**:
  - Key features included `CallDuration`, `Balance`, and `PreviousOutcome`.
  - Insights revealed the importance of contacting customers at optimal times and the relevance of their financial status.

### Visualizations
- Feature importance, ROC curves, and training/validation losses are provided for further analysis.

## Getting Started
1. **Install Python and Jupyter Notebook.**
2. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
3. **Run the Jupyter notebook**:
   ```bash
   jupyter notebook Capstone_final.ipynb
   ```
4. **Install dependencies**: Required packages will install automatically when running the notebook.

## Future Work
- **Enhanced Data Imputation**: Use predictive algorithms for handling missing values.
- **Additional Models**: Experiment with other machine learning algorithms to capture different patterns.
- **Further Optimization**: Explore more hyperparameters and configurations for existing models.
- **Expand `insurance-predictor` Package**: Add functions for raw data manipulation and model customization.

## References
1. Nyce, C. & Cpcu, A. (2007). *Predictive analytics white paper*. American Institute for CPCU.
2. Muley, R. (2018). *Data analytics for the insurance industry: A gold mine*. Journal of the Insurance Institute of India.
3. Hanafy, M. & Ming, R. (2021). *Machine learning approaches for auto insurance big data*. *Risks*, 9(2).
4. Ejiyi, C.J., et al. (2022). *Comparative analysis of building insurance prediction using some machine learning algorithms*.

