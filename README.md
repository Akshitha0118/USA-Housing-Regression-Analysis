# USA-Housing-Regression-Analysis

# 🏡 USA Housing Price Prediction Using Machine Learning

This project implements and compares **multiple machine learning regression models** to predict house prices using the **USA Housing dataset**.  
The goal is to evaluate different algorithms and identify the best-performing model based on standard regression metrics.

---

## 🚀 Project Overview

Accurate house price prediction is crucial for real estate analytics.  
This project explores a wide range of **linear, ensemble, neural network, and boosting models**, trains them on the same dataset, evaluates their performance, and saves each trained model for future use.

---

## 🧠 Machine Learning Models Implemented

- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- Elastic Net  
- Robust Regression (Huber)  
- Polynomial Regression  
- Stochastic Gradient Descent (SGD)  
- K-Nearest Neighbors (KNN)  
- Support Vector Regression (SVR)  
- Random Forest Regressor  
- Artificial Neural Network (MLPRegressor)  
- LightGBM Regressor  
- XGBoost Regressor  

---

## 🛠️ Technologies & Libraries

- **Python**
- **Pandas & NumPy**
- **Scikit-learn**
- **LightGBM**
- **XGBoost**
- **Pickle** (Model Serialization)

---

## 🔄 Workflow

1. Load the USA Housing dataset  
2. Preprocess data (feature selection & target separation)  
3. Split data into training and testing sets  
4. Train multiple regression models  
5. Evaluate models using:
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - R² Score  
6. Save trained models as `.pkl` files  
7. Store evaluation results in a CSV file  

---

## 📊 Model Evaluation Metrics

Each model is evaluated using:
- **MAE (Mean Absolute Error)**
- **MSE (Mean Squared Error)**
- **R² Score**

