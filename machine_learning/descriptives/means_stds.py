import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from constants import ROOT_DIR

df = pd.read_csv(os.path.join(ROOT_DIR, "data/out/openface_data.csv"))

mean_cols = [col for col in df.columns if "mean" in col.lower()]

scaler = MinMaxScaler()
df[mean_cols] = scaler.fit_transform(df[mean_cols])

# Compute mean and std per condition and emotion
summary_stats = df.groupby(['condition', 'emotion'])[mean_cols].agg(['mean', 'std']).reset_index()

# Plot feature distributions per condition
for feature in mean_cols[:5]:  # Plot first 5 features as example
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x=feature, hue="condition", common_norm=False)
    plt.title(f"Feature Distribution: {feature}")
    plt.show()

for feature in mean_cols[:5]:  # Plot first 5 features as example
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x=feature, hue="emotion", common_norm=False)
    plt.title(f"Feature Distribution by Emotion: {feature}")
    plt.show()