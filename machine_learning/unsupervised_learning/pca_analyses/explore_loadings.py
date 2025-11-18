import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from constants import ROOT_DIR
from machine_learning.utils import functional_scale_by

# ---------------- Config ----------------
FAMILY = "all"   # options: "pose", "gaze", "AU", "all"
TOP_N  = 10       # top loadings to show per PC
POINT_SIZE = 40

pattern_map = {
    "pose": r"pose.*mean",
    "gaze": r"gaze.*mean",
    "AU":   r"AU.*mean",
    "all":  r"mean"
}
pattern = pattern_map[FAMILY]

# ---------------- Load & select ----------------
openface_path = os.path.join(ROOT_DIR, "data/out/openface_data.csv")
df = pd.read_csv(openface_path)

feat_cols = df.columns[df.columns.str.contains(pattern, regex=True)]
needed = ['condition', 'video_id']
df = df.dropna(subset=needed + feat_cols.tolist())

conditions = df['condition'].values
video_ids  = df['video_id'].values
X = df.loc[:, feat_cols].values

# ---------------- Normalize within actor ----------------
X_norm = functional_scale_by(X, video_ids, "standard")

# ---------------- PCA ----------------
pca = PCA(n_components=2)
scores = pca.fit_transform(X_norm)  # shape (N, 2)
expl_var = pca.explained_variance_ratio_ * 100

# ---------------- Loadings ----------------
loadings = pd.DataFrame(
    pca.components_.T,                 # (n_features, n_components)
    index=feat_cols,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)]
)

# Top contributors (by absolute loading)
top_pc1 = loadings["PC1"].abs().sort_values(ascending=False).head(TOP_N).index
top_pc2 = loadings["PC2"].abs().sort_values(ascending=False).head(TOP_N).index

print(f"Using {FAMILY} features: {len(feat_cols)} columns")
print("\nTop contributors to PC1:")
print(loadings.loc[top_pc1, "PC1"].sort_values(key=np.abs, ascending=False))
print("\nTop contributors to PC2:")
print(loadings.loc[top_pc2, "PC2"].sort_values(key=np.abs, ascending=False))

# ---------------- Plots ----------------
plt.rcParams.update({"font.size": 14})

# 1) Scores scatter by condition (distinct markers, subtle grid)
plt.figure(figsize=(8, 6))
unique_conds = np.unique(conditions)
markers = ["o", "s", "^", "D", "v", "P", "X", "*"]  # rotate if >8 conditions

for i, cond in enumerate(unique_conds):
    mask = (conditions == cond)
    plt.scatter(
        scores[mask, 0],
        scores[mask, 1],
        label=str(cond),
        alpha=0.7,
        s=POINT_SIZE,
        marker=markers[i % len(markers)],
    )

plt.xlabel(f"PC1 ({expl_var[0]:.1f}%)")
plt.ylabel(f"PC2 ({expl_var[1]:.1f}%)")
# plt.title(f"PCA scores by embodiment ({FAMILY} features)")
# plt.title(f"PCA scores by embodiment")

plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(loc="best", fontsize=12)
plt.tight_layout()
plt.savefig(f"pca_scores_by_condition_{FAMILY}.png", dpi=200)
plt.show()

# # 2) Loading bars for PC1 and PC2 (top N features)
# def plot_top_loadings(pc_name, top_idx):
#     vals = loadings.loc[top_idx, pc_name].sort_values(key=np.abs)
#     plt.figure(figsize=(8, 6))
#     vals.plot(kind="barh")
#     plt.title(f"Top {TOP_N} loadings for {pc_name} ({FAMILY})")
#     plt.xlabel("Loading")
#     plt.grid(True, linestyle="--", alpha=0.3)
#     plt.tight_layout()
#     # plt.savefig(f"loadings_{pc_name.lower()}_{FAMILY}.png", dpi=200)
#     plt.show()
#
# plot_top_loadings("PC1", top_pc1)
# plot_top_loadings("PC2", top_pc2)
