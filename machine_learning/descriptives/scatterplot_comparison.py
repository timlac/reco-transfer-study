import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from nexa_py_sentimotion_mapper.sentimotion_mapper import Mapper
from sklearn.preprocessing import StandardScaler
from constants import AU_INTENSITY_COLS, ROOT_DIR, var_name_to_feat

# Load your CSV file into a DataFrame
df = pd.read_csv(os.path.join(ROOT_DIR, "data/out/openface_data.csv"))

feature = "AU04_r_mean"

# Split dataframe by condition
orig = df[df["condition"] == "original"][feature].reset_index(drop=True)
furhat = df[df["condition"] == "furhat"][feature].reset_index(drop=True)
meta = df[df["condition"] == "metahuman"][feature].reset_index(drop=True)

plt.figure(figsize=(10, 5))

# Original vs Furhat
plt.subplot(1, 2, 1)
plt.scatter(orig, furhat, alpha=0.7)
plt.plot([orig.min(), orig.max()], [furhat.min(), furhat.max()], 'r--')
plt.xlabel("Original", fontsize=14)  # Increased font size
plt.ylabel("Furhat", fontsize=14)  # Increased font size
plt.tick_params(axis='both', labelsize=12)  # Increased tick font size
plt.title(f"{var_name_to_feat[feature]} vs. Furhat", fontsize=14)  # Merged headline

# Original vs MetaHuman
plt.subplot(1, 2, 2)
plt.scatter(orig, meta, alpha=0.7)
plt.plot([orig.min(), orig.max()], [meta.min(), meta.max()], 'r--')
plt.xlabel("Original", fontsize=14)  # Increased font size
plt.ylabel("MetaHuman", fontsize=14)  # Increased font size
plt.tick_params(axis='both', labelsize=12)  # Increased tick font size
plt.title(f"{var_name_to_feat[feature]} vs. MetaHuman", fontsize=14)  # Merged headline

plt.tight_layout()
plt.show()