import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler

from constants import AU_INTENSITY_COLS, ROOT_DIR

# Load your CSV file into a DataFrame
df_openface = pd.read_csv(os.path.join(ROOT_DIR, "data/out/openface_data.csv"))

df = df_openface[df_openface["condition"] == "original"]

emotions = df["emotion"]


mean_cols = [col for col in df.columns if "mean" in col.lower()]

features = df.loc[:, df.columns.str.contains('mean')]

# Normalize the vectors
scaler = StandardScaler()
vec_normalized = scaler.fit_transform(features)

