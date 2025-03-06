import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from sklearn.preprocessing import MinMaxScaler

from constants import AU_INTENSITY_COLS, ROOT_DIR

# Load your CSV file
df = pd.read_csv(os.path.join(ROOT_DIR, "data/out/openface_data.csv"))

# Select feature columns dynamically
mean_cols = [col for col in df.columns if "mean" in col.lower()]

scaler = MinMaxScaler()
df[mean_cols] = scaler.fit_transform(df[mean_cols])

# Compute variance for each feature within each condition
variance_df = df.groupby('condition')[mean_cols].var().T

# Plot the variance for each condition
plt.figure(figsize=(12, 6))
variance_df.plot(kind='bar', figsize=(15, 6))
plt.xticks(rotation=45, ha='right')
plt.title("Feature Variance per Condition")
plt.ylabel("Variance")
plt.xlabel("Feature")
plt.legend(title="Condition")
plt.tight_layout()
plt.show()
