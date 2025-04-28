# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Load and preprocess the data
df = pd.read_csv("C:/Users/STEFANA DCRUZ/Downloads/Housing.csv")

# Encode 'yes'/'no' to 1/0 for binary columns
df.replace({'yes': 1, 'no': 0}, inplace=True)

# One-Hot Encode the 'furnishingstatus' column
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

# Optional: check if any missing values
# print(df.isnull().sum())

# Step 3: Feature Selection
# Selecting initial and additional features
features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 
            'airconditioning', 'prefarea', 
            'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished']

X = df[features]
y = df['price']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Fitting
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 8: Interpret Coefficients
coefficients = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
print("\nFeature Coefficients:")
print(coefficients)

# Step 9: Plotting
plt.figure(figsize=(8,6))
plt.scatter(X_test['area'], y_test, color='blue', label='Actual Price', alpha=0.6)
plt.scatter(X_test['area'], y_pred, color='red', label='Predicted Price', alpha=0.6)
plt.xlabel('Area (sq ft)')
plt.ylabel('Price')
plt.title('Actual vs Predicted Housing Prices (Based on Area)')
plt.legend()
plt.show()
