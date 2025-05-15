import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load cleaned dataset (from Algorithm 3)
cleaned_data = pd.read_csv('pid_dataset_cleaned(0.2).csv')  # Columns: ['Tr', 'Ts', 'Overshoot', 'Peak', 'kp', 'ki', 'kd']

# Split predictors (X) and responses (y)
X = cleaned_data[['Tr', 'Ts', 'Overshoot', 'Peak']]
y_kp = cleaned_data['kp']
y_ki = cleaned_data['ki']
y_kd = cleaned_data['kd']

# Split data (80% train, 20% test)
X_train, X_test, y_kp_train, y_kp_test = train_test_split(X, y_kp, test_size=0.2, random_state=42)
_, _, y_ki_train, y_ki_test = train_test_split(X, y_ki, test_size=0.2, random_state=42)
_, _, y_kd_train, y_kd_test = train_test_split(X, y_kd, test_size=0.2, random_state=42)

# Train models for kp, ki, kd
model_kp = LinearRegression().fit(X_train, y_kp_train)
model_ki = LinearRegression().fit(X_train, y_ki_train)
model_kd = LinearRegression().fit(X_train, y_kd_train)

# Predictions
y_kp_pred = model_kp.predict(X_test)
y_ki_pred = model_ki.predict(X_test)
y_kd_pred = model_kd.predict(X_test)

# Evaluate (MSE and R²)
def evaluate_model(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name}: MSE = {mse:.3f}, R² = {r2:.3f}")
    return mse, r2

mse_kp, r2_kp = evaluate_model(y_kp_test, y_kp_pred, "kp")
mse_ki, r2_ki = evaluate_model(y_ki_test, y_ki_pred, "ki")
mse_kd, r2_kd = evaluate_model(y_kd_test, y_kd_pred, "kd")

# Average performance (as in Table 4.1)
avg_mse = np.mean([mse_kp, mse_ki, mse_kd])
avg_r2 = np.mean([r2_kp, r2_ki, r2_kd])
print(f"\nAverage MSE: {avg_mse:.3f}, Average R²: {avg_r2:.3f}")

# Create a figure with subplots for predicted vs true values
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(y_kp_test, y_kp_pred, alpha=0.3)
plt.xlabel("True kp")
plt.ylabel("Predicted kp")
plt.title("kp: True vs Predicted")

plt.subplot(1, 3, 2)
plt.scatter(y_ki_test, y_ki_pred, alpha=0.3)
plt.xlabel("True ki")
plt.ylabel("Predicted ki")
plt.title("ki: True vs Predicted")

plt.subplot(1, 3, 3)
plt.scatter(y_kd_test, y_kd_pred, alpha=0.3)
plt.xlabel("True kd")
plt.ylabel("Predicted kd")
plt.title("kd: True vs Predicted")

plt.tight_layout()
plt.show()

# Calculate residuals
residuals_kp = y_kp_test - y_kp_pred
residuals_ki = y_ki_test - y_ki_pred
residuals_kd = y_kd_test - y_kd_pred

# Create a figure with subplots for residual distributions
plt.figure(figsize=(15, 5))

# kp residuals
plt.subplot(1, 3, 1)
plt.hist(residuals_kp, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('kp Residual Distribution')
plt.axvline(x=0, color='r', linestyle='--')

# ki residuals
plt.subplot(1, 3, 2)
plt.hist(residuals_ki, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('ki Residual Distribution')
plt.axvline(x=0, color='r', linestyle='--')

# kd residuals
plt.subplot(1, 3, 3)
plt.hist(residuals_kd, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('kd Residual Distribution')
plt.axvline(x=0, color='r', linestyle='--')

plt.tight_layout()
plt.show()
