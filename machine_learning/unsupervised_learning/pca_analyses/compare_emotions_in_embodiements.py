import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from machine_learning.utils import functional_scale_by
from constants import ROOT_DIR
from sklearn.preprocessing import StandardScaler, MinMaxScaler

openface_path = os.path.join(ROOT_DIR, "data/out/openface_data.csv")
df = pd.read_csv(openface_path)

# Keep only rows with required fields and feature columns
feat_cols = df.columns[df.columns.str.contains('mean')]
# feat_cols = df.columns[df.columns.str.contains("mean") & df.columns.str.contains("AU")]
# feat_cols = df.columns[df.columns.str.contains("mean") & df.columns.str.contains("gaze")]
# feat_cols = df.columns[df.columns.str.contains("mean") & df.columns.str.contains("pose")]

needed = ['condition', 'video_id']
df = df.dropna(subset=needed + feat_cols.tolist())

# Meta (as arrays, no DataFrame)
conditions = df['condition'].values
df = df[df['condition'] == "furhat"]  # filter to single embodiment for emotion comparison


video_ids  = df['video_id'].values
emotions = df['emotion'].values




# Features
vec = df.loc[:, feat_cols].values

scaler = StandardScaler()

# Normalize per video_id (standardize within each video)
vec_normalized = functional_scale_by(vec, video_ids, "standard")
# vec_normalized = functional_scale_by(vec, conditions, "standard")
# vec_normalized = scaler.fit_transform(vec)

# PCA -> 2D
pca = PCA(n_components=2)
xy = pca.fit_transform(vec_normalized)  # shape (N, 2)

# Plot settings
plt.figure(figsize=(8, 6))
plt.rcParams.update({"font.size": 14})  # larger font

unique_emos = np.unique(emotions)
markers = ["o", "s", "^", "D", "v", "P", "X", "*"]  # enough unique markers

for i, emo in enumerate(unique_emos):
    mask = (emotions == emo)
    plt.scatter(
        xy[mask, 0],
        xy[mask, 1],
        label=str(emo),
        alpha=0.7,
        s=40,
        marker=markers[i % len(markers)]
    )

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("PCA Visual Features")
plt.legend(loc="best", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.3)  # subtle grid
plt.tight_layout()
plt.show()
