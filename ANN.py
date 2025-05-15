import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load cleaned dataset
data = pd.read_csv('pid_dataset_cleaned(0.2).csv')  # Columns: ['Tr', 'Ts', 'Overshoot', 'Peak', 'kp', 'ki', 'kd']

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(data[['Tr', 'Ts', 'Overshoot', 'Peak']])
y = data[['kp', 'ki', 'kd']].values

# Split data (90% train, 10% test - matches dissertation Section 5.3.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)


# ANN Model (2 hidden layers)
class PID_ANN(nn.Module):
    def __init__(self, input_size=4, hidden1=14, hidden2=10, output_size=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.output = nn.Linear(hidden2, output_size)
        self.sigmoid = nn.Sigmoid()  # Matches dissertation's sigmoidal activation

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.output(x)
        return x


# Initialize model (architecture from dissertation Table 5.2 for σ²=5.0)
model = PID_ANN(hidden1=14, hidden2=10)  # Architecture for ψ₆ (high variance dataset)

# Loss and optimizer (Levenberg-Marquardt approximation)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam approximates LM behavior

# Initialize lists to store losses
train_losses = []
test_losses = []

# Training loop
epochs = 3000
for epoch in range(epochs):
    # Training phase
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # Evaluation phase
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses.append(test_loss.item())

    if (epoch + 1) % 50 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# Plot training and test losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Test Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Final Evaluation
with torch.no_grad():
    y_pred = model(X_test)

    # Calculate overall MSE
    mse = criterion(y_pred, y_test).item()

    # Calculate individual MSEs for each gain
    mse_kp = torch.mean((y_test[:, 0] - y_pred[:, 0]) ** 2).item()
    mse_ki = torch.mean((y_test[:, 1] - y_pred[:, 1]) ** 2).item()
    mse_kd = torch.mean((y_test[:, 2] - y_pred[:, 2]) ** 2).item()

    y_var = torch.var(y_test, unbiased=False).item()
    nmse = mse / y_var

    print(f"\nFinal Test MSE: {mse:.4f}, Normalized MSE: {nmse:.4f}")
    print(f"Individual MSEs: kp={mse_kp:.4f}, ki={mse_ki:.4f}, kd={mse_kd:.4f}")


# Calculate R² scores
def r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


r2_kp = r2_score(y_test[:, 0], y_pred[:, 0])
r2_ki = r2_score(y_test[:, 1], y_pred[:, 1])
r2_kd = r2_score(y_test[:, 2], y_pred[:, 2])
print(f"R² Scores: kp={r2_kp:.3f}, ki={r2_ki:.3f}, kd={r2_kd:.3f}")


# --- Residual Distribution Plots ---
def plot_residual_distributions(y_true, y_pred):
    """Generate histogram plots of residual distributions for kp, ki, kd"""
    gains = ['kp', 'ki', 'kd']
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()

    plt.figure(figsize=(18, 6))

    for i, gain in enumerate(gains):
        residuals = y_true_np[:, i] - y_pred_np[:, i]

        plt.subplot(1, 3, i + 1)
        plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--', label='Zero error')
        plt.xlabel('Residual (True - Predicted)')
        plt.ylabel('Frequency')
        plt.title(f'{gain} Residual Distribution\n(MSE={eval(f"mse_{gain}"):.4f}, Std={np.std(residuals):.3f})')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


# --- Prediction Visualization ---
def evaluate_predictions(y_true, y_pred):
    """Generate predicted vs. true and residual plots with threshold metrics."""
    gains = ['kp', 'ki', 'kd']
    thresholds = {
        'PERFECT': 0.05,  # ±0.05 (matches dissertation)
        'EXCELLENT': 0.10,
        'GOOD': 0.25
    }

    # Convert tensors to numpy for plotting
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()

    # 1. Predicted vs. True Values
    plt.figure(figsize=(18, 6))
    for i, gain in enumerate(gains):
        plt.subplot(1, 3, i + 1)
        plt.scatter(y_true_np[:, i], y_pred_np[:, i], alpha=0.5, label=f'{gain} predictions')
        plt.plot([y_true_np[:, i].min(), y_true_np[:, i].max()],
                 [y_true_np[:, i].min(), y_true_np[:, i].max()],
                 'r--', label='Perfect prediction')
        plt.xlabel(f'True {gain}')
        plt.ylabel(f'Predicted {gain}')
        plt.title(
            f'{gain}: Predicted vs. True (MSE={eval(f"mse_{gain}"):.4f}, R²={r2_score(y_true[:, i], y_pred[:, i]):.3f})')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 2. Residual Plots with Thresholds
    plt.figure(figsize=(18, 6))
    for i, gain in enumerate(gains):
        residuals = y_true_np[:, i] - y_pred_np[:, i]

        plt.subplot(1, 3, i + 1)
        plt.scatter(y_pred_np[:, i], residuals, alpha=0.5, label='Residuals')
        plt.axhline(y=0, color='r', linestyle='--', label='Zero error')

        # Add threshold bands (dissertation-style)
        colors = {'GOOD': 'green', 'EXCELLENT': 'blue', 'PERFECT': 'purple'}
        for level, thresh in thresholds.items():
            plt.axhspan(-thresh, thresh, alpha=0.1, color=colors[level], label=f'±{thresh} ({level})')

        plt.xlabel(f'Predicted {gain}')
        plt.ylabel('Residual (True - Predicted)')
        plt.title(f'{gain}: Residuals (MSE={eval(f"mse_{gain}"):.4f})')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 3. Calculate threshold metrics (dissertation Table 5.4)
    print("\nPrediction Certainty Metrics:")
    for gain in gains:
        residuals = y_true_np[:, gains.index(gain)] - y_pred_np[:, gains.index(gain)]
        print(f"\n{gain} (MSE={eval(f'mse_{gain}'):.4f}):")
        for level, thresh in thresholds.items():
            within_thresh = np.sum(np.abs(residuals) <= thresh) / len(residuals) * 100
            print(f"{level}: {within_thresh:.1f}% of predictions within ±{thresh}")


# Generate all plots and metrics
plot_residual_distributions(y_test, y_pred)
evaluate_predictions(y_test, y_pred)