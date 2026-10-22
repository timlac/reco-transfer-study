import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from constants import AU_INTENSITY_COLS, au_brief_names


BASE = "/home/tim/Work/recordix/reco-transfer-study/data/visualization_example/A435_conf_v_3"
CSV_DIR = f"{BASE}/openface_2"

FILES = {
    "Original":   "original_A435_conf_v_3.csv",
    "Furhat":     "furhat_A435_conf_v_3.csv",
    "Metahuman":  "metahuman_A435_conf_v_3.csv",
}


# ------------------------
# LOAD + NORMALIZE AUs
# ------------------------
def load_normalized_aus(csv_path, frame_idx):
    df = pd.read_csv(csv_path)

    row = df[df["frame"] == frame_idx]
    if len(row) == 0:
        row = df.iloc[[0]]

    row = row.iloc[0]

    norm_vals = {}
    for au in AU_INTENSITY_COLS:
        mn = df[au].min()
        mx = df[au].max()
        norm_vals[au] = (row[au] - mn) / (mx - mn + 1e-8)

    return norm_vals


# ------------------------
# RADAR PLOT (FILTERED)
# ------------------------
def radar_plot_filtered(frame_orig, frame_fur, frame_meta):

    # Load all AU values
    orig = load_normalized_aus(os.path.join(CSV_DIR, FILES["Original"]), frame_orig)
    fur  = load_normalized_aus(os.path.join(CSV_DIR, FILES["Furhat"]),   frame_fur)
    meta = load_normalized_aus(os.path.join(CSV_DIR, FILES["Metahuman"]), frame_meta)

    # ------------------------
    # FILTER: keep only AUs where ANY condition > 0
    # ------------------------
    filtered_aus = [
        au for au in AU_INTENSITY_COLS
        if not (orig[au] == 0 and fur[au] == 0 and meta[au] == 0)
    ]

    labels = [au_brief_names[a] for a in filtered_aus]

    # Values (close polygon by repeating first element)
    orig_vals = np.array([orig[a] for a in filtered_aus])
    fur_vals  = np.array([fur[a]  for a in filtered_aus])
    meta_vals = np.array([meta[a] for a in filtered_aus])

    orig_vals = np.concatenate([orig_vals, [orig_vals[0]]])
    fur_vals  = np.concatenate([fur_vals,  [fur_vals[0]]])
    meta_vals = np.concatenate([meta_vals, [meta_vals[0]]])

    angles = np.linspace(0, 2 * np.pi, len(filtered_aus) + 1)

    # ------------------------
    # Plot
    # ------------------------
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)

    ax.plot(angles, orig_vals, linewidth=2, label="Original")
    ax.fill(angles, orig_vals, alpha=0.15)

    ax.plot(angles, fur_vals, linewidth=2, label="Furhat")
    ax.fill(angles, fur_vals, alpha=0.15)

    ax.plot(angles, meta_vals, linewidth=2, label="Metahuman")
    ax.fill(angles, meta_vals, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)

    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8"], fontsize=8)

    ax.set_title("Filtered Radar Plot — AUs Active in At Least One Condition", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    plt.tight_layout()
    plt.show()


# ------------------------
# Example call
# ------------------------
radar_plot_filtered(
    frame_orig=110,
    frame_fur=110,
    frame_meta=110
)
