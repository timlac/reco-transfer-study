import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from nexa_py_sentimotion_mapper.sentimotion_mapper import Mapper


from sklearn.preprocessing import StandardScaler

from constants import AU_INTENSITY_COLS, ROOT_DIR

# Load your CSV file into a DataFrame
df_openface = pd.read_csv(os.path.join(ROOT_DIR, "data/out/openface_data.csv"))

df = df_openface[df_openface["condition"] == "metahuman"].copy()

emotions = df["emotion"]
emotion_ids = df["emotion_id"]

mean_cols = [col for col in df.columns if "mean" in col.lower()]

print(mean_cols)

df[mean_cols] = StandardScaler().fit_transform(df[mean_cols])

for feature in mean_cols:  # Plot first 5 features as example
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x=feature, hue="emotion", common_norm=False)
    plt.title(f"Feature Distribution by Emotion: {feature}")
    plt.show()