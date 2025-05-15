import numpy as np
import pandas as pd
from scipy.stats import zscore
from control import tf, feedback, series, step_response


# 1. Your original data generation (with built-in feature extraction)
def generate_synthetic_data(n_samples=40000, variance=3):
    system = tf([1], [1, 3, 1])  # G(s) = 1/(s² + 3s + 1)
    kp_opt, ki_opt, kd_opt = 11.15, 7.86, 2.86  # Optimal gains (Table 3.1)

    data = []
    for _ in range(n_samples):
        kp = np.clip(kp_opt + variance * np.random.randn(), 8, 14)
        ki = np.clip(ki_opt + variance * np.random.randn(), 5, 10)
        kd = np.clip(kd_opt + variance * np.random.randn(), 1, 5)

        pid = tf([kd, kp, ki], [1, 0])  # PID controller
        open_loop = series(pid, system)
        closed_loop = feedback(open_loop)

        try:
            t, y = step_response(closed_loop, T=np.linspace(0, 25, 1000))
            if np.all(np.isfinite(y)):
                y_final = y[-1]
                y_max = np.max(y)
                overshoot = 100 * (y_max - y_final) / y_final if y_final != 0 else 0

                # Rise Time (10% to 90%)
                idx_10 = np.argmax(y >= 0.1 * y_final)
                idx_90 = np.argmax(y >= 0.9 * y_final)
                rise_time = t[idx_90] - t[idx_10]

                # Settling Time (within 2%)
                settling_idx = np.where(np.abs(y - y_final) <= 0.02 * y_final)[0]
                settling_time = t[settling_idx[0]] if len(settling_idx) > 0 else t[-1]

                data.append([rise_time, settling_time, overshoot, y_max, kp, ki, kd])
        except:
            continue

    return pd.DataFrame(
        data,
        columns=['Tr', 'Ts', 'Overshoot', 'Peak', 'kp', 'ki', 'kd']
    )


# 2. Algorithm 3 (Cleaning)
def clean_dataset(df, z_threshold=4.0):
    """Remove outliers using Z-score on metrics."""
    predictors = df[['Tr', 'Ts', 'Overshoot', 'Peak']]
    z_scores = predictors.apply(zscore)
    outliers = (np.abs(z_scores) > z_threshold).any(axis=1)
    cleaned_df = df[~outliers].copy()
    print(f"Removed {outliers.sum()} outliers (Z-score > {z_threshold})")
    return cleaned_df


# Generate and clean data
raw_data = generate_synthetic_data(n_samples=40000, variance=3)
cleaned_data = clean_dataset(raw_data)

# Save to CSV (ready for ML)
cleaned_data.to_csv('pid_dataset_cleaned(0.2).csv', index=False)
print("Final dataset shape:", cleaned_data.shape)