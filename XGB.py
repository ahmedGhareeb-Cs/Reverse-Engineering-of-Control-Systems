import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# ======================
# DATA PREPARATION
# ======================
def load_and_preprocess_data():
    """Load and preprocess the PID dataset"""
    data = pd.read_csv('pid_dataset_cleaned(0.2).csv')

    # Feature engineering
    data['Tr_Ts_ratio'] = data['Tr'] / (data['Ts'] + 1e-6)
    data['Overshoot_Peak'] = data['Overshoot'] * data['Peak']

    feature_names = ['Tr', 'Ts', 'Overshoot', 'Peak', 'Tr_Ts_ratio', 'Overshoot_Peak']
    X = data[feature_names].values
    y = data[['kp', 'ki', 'kd']].values

    # Train-test split (90-10)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, feature_names


# ======================
# HYPERPARAMETER TUNING (OPTIMIZED)
# ======================
def tune_hyperparameters(X_train, y_train):
    """Optimized hyperparameter tuning with reduced search space"""
    param_grid = {
        'n_estimators': [100, 500],
        'max_depth': [4, 6],
        'learning_rate': [0.03, 0.05],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [0, 0.1]
    }

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        eval_metric='rmse',
        random_state=42,
        n_jobs=-1
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=KFold(n_splits=2, shuffle=True, random_state=42),
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


# ======================
# FINAL MODEL TRAINING
# ======================
def train_final_model(X_train, y_train, best_params):
    """Train final model with best parameters and early stopping"""
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    final_params = best_params.copy()
    final_params['n_estimators'] = 1000

    model = xgb.XGBRegressor(
        **final_params,
        objective='reg:squarederror',
        tree_method='hist',
        eval_metric='rmse',
        early_stopping_rounds=20,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10
    )

    return model


# ======================
# VISUALIZATION
# ======================
def plot_results(y_test, y_pred, feature_names):
    """Generate evaluation plots with feature names"""
    gains = ['kp', 'ki', 'kd']

    # Individual metrics for each gain
    print("\n=== Individual Gain Metrics ===")
    for i, gain in enumerate(gains):
        mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        print(f"{gain}: MSE = {mse:.4f}, R² = {r2:.3f}")

    # 1. Prediction vs Actual
    plt.figure(figsize=(18, 5))
    for i, gain in enumerate(gains):
        plt.subplot(1, 3, i + 1)
        sns.regplot(x=y_test[:, i], y=y_pred[:, i], scatter_kws={'alpha': 0.4})
        plt.plot([y_test[:, i].min(), y_test[:, i].max()],
                 [y_test[:, i].min(), y_test[:, i].max()], 'r--')
        plt.xlabel(f'True {gain}')
        plt.ylabel(f'Predicted {gain}')
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        plt.title(f'{gain} (R²={r2:.3f})')
    plt.tight_layout()
    plt.show()

    # 2. Residual Analysis
    plt.figure(figsize=(18, 5))
    for i, gain in enumerate(gains):
        residuals = y_test[:, i] - y_pred[:, i]
        plt.subplot(1, 3, i + 1)
        sns.histplot(residuals, kde=True, bins=30)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel(f'Residual ({gain})')
        mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        plt.title(f'{gain} (MSE={mse:.4f})')
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names):
    """Plot feature importance with proper names"""
    importance = model.get_booster().get_score(importance_type='weight')

    # Create mapping from feature index to name
    importance_mapped = {feature_names[int(k[1:])]: v for k, v in importance.items()}

    # Sort features by importance
    importance_sorted = dict(sorted(importance_mapped.items(), key=lambda item: item[1], reverse=True))

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(importance_sorted.values()),
                y=list(importance_sorted.keys()),
                palette='viridis')
    plt.title('Feature Importance (Weight)')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()


# ======================
# MAIN EXECUTION
# ======================
def main():
    # Load data
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data()

    # Hyperparameter tuning
    print("Starting optimized hyperparameter tuning...")
    best_model, best_params = tune_hyperparameters(X_train, y_train)
    print("\n=== Best Parameters ===")
    for k, v in best_params.items():
        print(f"{k}: {v}")

    # Train final model with early stopping
    print("\nTraining final model with early stopping...")
    final_model = train_final_model(X_train, y_train, best_params)

    # Predictions
    y_pred = final_model.predict(X_test)

    # Evaluation
    print("\n=== Final Evaluation ===")
    print(f"Overall Test MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"Overall Test R²: {r2_score(y_test, y_pred):.3f}")

    # Feature importance with names
    plot_feature_importance(final_model, feature_names)

    # Detailed plots with individual metrics
    plot_results(y_test, y_pred, feature_names)

    # Save model
    final_model.save_model('pid_xgboost_tuned_model.json')
    print("Model saved to pid_xgboost_tuned_model.json")


if __name__ == "__main__":
    main()