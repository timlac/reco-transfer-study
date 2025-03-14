import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from nexa_py_sentimotion_mapper.sentimotion_mapper import Mapper


from sklearn.preprocessing import StandardScaler

from constants import AU_INTENSITY_COLS, ROOT_DIR

# Load your CSV file into a DataFrame
df_openface = pd.read_csv(os.path.join(ROOT_DIR, "data/out/openface_data.csv"))

df = df_openface[df_openface["condition"] == "original"].copy()

emotions = df["emotion"]
emotion_ids = df["emotion_id"]

mean_cols = [col for col in df.columns if "mean" in col.lower()]

print(mean_cols)

df[mean_cols] = StandardScaler().fit_transform(df[mean_cols])

for idx in range(0, len(mean_cols)):

    if idx == 1:
        continue

    for emotion_id in emotion_ids.unique():
        df_emotion = df[df["emotion_id"] == emotion_id]

        x, y = df_emotion[mean_cols[idx]], df_emotion[mean_cols[1]]
        plt.scatter(x, y)

    plt.legend(Mapper.get_emotion_from_id(emotion_ids.unique()), bbox_to_anchor=(1, 0.5))
    plt.xlabel(mean_cols[idx])
    plt.ylabel(mean_cols[1])
    plt.title(f"Scatter plot of {mean_cols[idx]} and {mean_cols[1]}")
    plt.tight_layout()
    plt.show()