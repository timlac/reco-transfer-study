import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from constants import AU_INTENSITY_COLS

mean_cols = [f"{col}_mean" for col in AU_INTENSITY_COLS]
var_cols = [f"{col}_var" for col in AU_INTENSITY_COLS]

# Load your CSV file into a DataFrame
df = pd.read_csv("../../data/out/openface_data.csv")

# Melt the DataFrame for mean values
melted_mean_df = df.melt(id_vars=['condition', 'emotion'],
                         value_vars=mean_cols,
                         var_name='Action Unit',
                         value_name='Mean Intensity')

# Adjust the Action Unit column to remove the "_mean" suffix
melted_mean_df['Action Unit'] = melted_mean_df['Action Unit'].str.replace('_mean', '')

# Melt the DataFrame for variance values
melted_var_df = df.melt(id_vars=['condition', 'emotion'],
                        value_vars=var_cols,
                        var_name='Action Unit',
                        value_name='Variance')

# Adjust the Action Unit column to remove the "_var" suffix
melted_var_df['Action Unit'] = melted_var_df['Action Unit'].str.replace('_var', '')

# Now you can plot separately for mean and variance
emotions = df['emotion'].unique()

# Plotting mean intensities
for emotion in emotions:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted_mean_df[melted_mean_df['emotion'] == emotion],
                x='Action Unit',
                y='Mean Intensity',
                hue='condition')
    plt.title(f"Mean Action Unit Intensities for Emotion: {emotion}")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Action Unit")
    plt.ylabel("Mean Intensity")
    plt.legend(title="Condition")
    plt.tight_layout()
    plt.savefig(f"mean_intensity_{emotion}.png")
    plt.show()

# Plotting variances
for emotion in emotions:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted_var_df[melted_var_df['emotion'] == emotion],
                x='Action Unit',
                y='Variance',
                hue='condition')
    plt.title(f"Action Unit Variances for Emotion: {emotion}")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Action Unit")
    plt.ylabel("Variance")
    plt.legend(title="Condition")
    plt.tight_layout()
    # plt.savefig(f"variance_{emotion}.png")
    plt.show()