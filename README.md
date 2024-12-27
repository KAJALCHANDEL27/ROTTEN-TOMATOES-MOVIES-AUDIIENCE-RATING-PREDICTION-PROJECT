# ROTTEN-TOMATOES-MOVIES-AUDIIENCE-RATING-PREDICTION-PROJECT
This project focuses on predicting the audience rating of movies using various machine learning techniques, including linear regression, Ridge regression, and Lasso regression. The dataset used is the "Rotten Tomatoes Movies" dataset.
Table of Contents
1.	Import Libraries and Dataset Description
2.	Data Preprocessing
3.	Feature Engineering
4.	Model Training and Evaluation
5.	Hyperparameter Tuning
6.	Visualization
7.	Model Deployment
8.	Conclusion
9.	Future Work

Import necessary Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import pickle
Dataset Description
The dataset contains various features of movies, including their title, genre, rating, directors, cast, and audience rating. The primary goal is to predict the audience_rating.
Data Preprocessing
1.	Loading the Dataset:
dataset = pd.read_csv("Rotten_Tomatoes_Movies3.csv", encoding='latin1')

Display dataset information
print(dataset.head(5))
print(dataset.shape)
print(dataset.describe())
print(dataset.info())


2.	Handling Date Columns: Convert in_theaters_date to a Unix timestamp:

if 'in_theaters_date' in dataset.columns:
    dataset['in_theaters_date'] = pd.to_datetime(dataset['in_theaters_date'], errors='coerce', dayfirst=True)
    dataset['in_theaters_date'] = dataset['in_theaters_date'].view('int64') // 10**9




3.	Dropping Unnecessary Columns: Remove columns with the most null values:

columns_to_drop = ['critics_consensus', 'writers', 'studio_name', 'on_streaming_date', 'runtime_in_minutes', 'tomatometer_status', 'tomatometer_rating', 'tomatometer_count']
dataset.drop(columns=columns_to_drop, inplace=True)
dataset.dropna(inplace=True)

Feature Engineering
1.	One-Hot Encoding: Encode categorical columns into binary columns:

categorical_columns = ['movie_title', 'movie_info', 'rating', 'genre', 'directors', 'cast']
dataset_encoded = pd.get_dummies(dataset, columns=categorical_columns)

2.	Scaling Numerical Columns: Standardize the numerical features:

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_columns = dataset_encoded.select_dtypes(include=[float, int]).columns
dataset_encoded[numerical_columns] = scaler.fit_transform(dataset_encoded[numerical_columns])

Model Training and Evaluation
1.	Splitting the Dataset: Separate features and target, then split the data:

X = dataset_encoded.drop('audience_rating', axis=1)
y = dataset['audience_rating']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
2.	Dimensionality Reduction with PCA: Reduce the number of features:

from sklearn.decomposition import PCA
pca = PCA(n_components=100)
X_reduced = pca.fit_transform(X)
3.	Training Linear Regression Model:

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
4.	Evaluating the Model: Calculate MAE, MSE, and R²:

y_pred = model.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae}, MSE: {mse}, R²: {r2}")

Hyperparameter Tuning
1.	Ridge and Lasso Regression: Tune hyperparameters using GridSearchCV:

from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV

ridge = Ridge()
ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
lasso = Lasso()
lasso_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}

ridge_grid = GridSearchCV(ridge, ridge_params, cv=5, scoring='r2')
ridge_grid.fit(X_train, y_train)

lasso_grid = GridSearchCV(lasso, lasso_params, cv=5, scoring='r2')
lasso_grid.fit(X_train, y_train)

best_ridge = ridge_grid.best_estimator_
best_lasso = lasso_grid.best_estimator_

2.	Evaluating Ridge and Lasso Models:

y_pred_ridge = best_ridge.predict(X_test)
y_pred_lasso = best_lasso.predict(X_test)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
print(f"Ridge - MAE: {mae_ridge}, MSE: {mse_ridge}, R²: {r2_ridge}")
print(f"Lasso - MAE: {mae_lasso}, MSE: {mse_lasso}, R²: {r2_lasso}")

Visualization:
•	Scatter plot of actual vs. predicted values: The scatter plot of actual vs. predicted values helps you see how close your predictions are to the actual values. Ideally, the points should lie along the red line, indicating perfect predictions.
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual vs. Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Line of perfect prediction
plt.show()

•	Residual plot : The residual plot shows the differences between the actual and predicted values. Ideally, residuals should be randomly scattered around zero, indicating no obvious patterns.
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), color='red')  # Horizontal line at 0
plt.show()


•	Distribution plot of residuals : This plot helps you check if the residuals follow a normal distribution, which is an assumption of linear regression.
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

Model Deployment
1.	Saving the Model:
import pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
2.	Creating Flask API:
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict(np.array([data['features']]))
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=False)

 Conclusion

In this project, we developed and evaluated a machine learning model to predict audience ratings for movies using a dataset from Rotten Tomatoes. We followed a comprehensive pipeline that included:

1. Data Preprocessing:
•	Loading and inspecting the dataset.
•	Handling missing values by dropping columns with a high percentage of null values and rows with any null values.
•	Converting date columns to a numerical format for easier handling.

2. Feature Engineering:
•	One-hot encoding categorical features to convert them into a numerical format suitable for machine learning algorithms.

3. Scaling and Dimensionality Reduction:
•	Standardizing numerical features to have zero mean and unit variance.
•	Reducing the dimensionality of the dataset using Principal Component Analysis (PCA) to retain the most informative features.

4. Model Training and Evaluation:
•	Training a Linear Regression model.
•	Evaluating the model using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² score.
•	Implementing hyperparameter tuning for Ridge and Lasso regression models to improve performance.



5. Visualization:
•	Creating scatter plots, residual plots, and distribution plots to visualize the model's performance and residuals.

Future Work

While the current model provides a good starting point for predicting movie audience ratings, there are several areas for future improvement:

1. Feature Selection:
•	Investigate additional features that might influence audience ratings, such as social media mentions, critic reviews, or audience demographics.

2. Advanced Models:
•	Experiment with more advanced machine learning models like Random Forest, Gradient Boosting, or Neural Networks to potentially improve prediction accuracy.

3. Hyperparameter Tuning:
•	Implement more sophisticated hyperparameter tuning techniques like Bayesian Optimization or Random Search to fine-tune model parameters.

4. Cross-Validation:
•	Use cross-validation to ensure the model's robustness and generalizability to new, unseen data.


5. Deployment:
•	Deploy the model in a production environment using a robust framework like Flask combined with a WSGI server like Gunicorn, ensuring it can handle real-world requests efficiently.

6. Real-Time Predictions:
•	Integrate the model into a web application or API to provide real-time predictions based on user input.

By addressing these areas, we can enhance the accuracy and usability of the model, providing more reliable predictions for movie audience ratings.
