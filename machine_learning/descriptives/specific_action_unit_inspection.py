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

df_furhat = df[df["condition"] == "furhat"]

print(df_furhat["AU10_r_mean"].value_counts())
print(df_furhat["AU06_r_mean"].value_counts())