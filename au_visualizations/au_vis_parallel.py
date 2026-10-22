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
        mn, mx = df[au].min(), df[au].max()
        norm_vals[au] = (row[au] - mn) / (mx - mn + 1e-8)

    return norm_vals


# ------------------------
# PARALLEL COORDINATES PLOT WITH FILL
# ------------------------
def plot_parallel_coordinates(frame_orig, frame_fur, frame_meta):

    aus = AU_INTENSITY_COLS

    # Load AU dicts
    orig_vals = load_normalized_aus(os.path.join(CSV_DIR, FILES["Original"]), frame_orig)
    fur_vals  = load_normalized_aus(os.path.join(CSV_DIR, FILES["Furhat"]),   frame_fur)
    meta_vals = load_normalized_aus(os.path.join(CSV_DIR, FILES["Metahuman"]), frame_meta)

    # Filter zero AUs
    filtered_aus = [
        au for au in aus
        if not (orig_vals[au] == 0 and fur_vals[au] == 0 and meta_vals[au] == 0)
    ]

    aus = filtered_aus
    labels = [au_brief_names[a] for a in aus]

    # Data matrix
    data = np.vstack([
        [orig_vals[a] for a in aus],
        [fur_vals[a]  for a in aus],
        [meta_vals[a] for a in aus]
    ])

    conditions = ["Original", "Furhat", "Metahuman"]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(aus))

    # Plot with fill
    for i, cond in enumerate(conditions):
        y = data[i]
        line = ax.plot(x, y, marker="o", linewidth=2, label=cond)[0]
        color = line.get_color()

        # fill between line and zero
        ax.fill_between(x, y, np.zeros_like(y), color=color, alpha=0.15)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized AU Intensity")
    ax.set_title("Parallel Coordinates Plot — Filtered AUs (with fill)")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()

    save_path = "parallel_coordinates_plot.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    print(f"Saved plot to: {save_path}")

    plt.show()



# ------------------------
# Example call
# ------------------------
plot_parallel_coordinates(
    frame_orig=110,
    frame_fur=110,
    frame_meta=110
)
