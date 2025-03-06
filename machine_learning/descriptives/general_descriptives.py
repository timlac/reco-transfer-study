import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from constants import AU_INTENSITY_COLS, ROOT_DIR

# Load your CSV file into a DataFrame
df = pd.read_csv(os.path.join(ROOT_DIR, "data/out/openface_data.csv"))

mean_cols = [col for col in df.columns if "mean" in col.lower()]

# Normalize mean intensity values using min-max scaling
df[mean_cols] = (df[mean_cols] - df[mean_cols].min()) / (df[mean_cols].max() - df[mean_cols].min())

# Melt the DataFrame for mean values
melted_mean_df = df.melt(id_vars=['condition', 'emotion'],
                         value_vars=mean_cols,
                         var_name='Feature',
                         value_name='Mean Intensity')

# Adjust the column to remove the "_mean" suffix
melted_mean_df['Feature'] = melted_mean_df['Feature'].str.replace('_mean', '')

# Now you can plot separately for mean and variance
emotions = df['emotion'].unique()

# Plotting mean intensities
for emotion in emotions:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted_mean_df[melted_mean_df['emotion'] == emotion],
                x='Feature',
                y='Mean Intensity',
                hue='condition')
    plt.title(f"Mean Feature Intensities for Emotion: {emotion}")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Feature")
    plt.ylabel("Mean Intensity")
    plt.legend(title="Condition")
    plt.tight_layout()
    # plt.savefig(f"mean_intensity_{emotion}.png")
    plt.show()