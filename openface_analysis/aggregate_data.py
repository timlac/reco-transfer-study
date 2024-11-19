import pandas as pd
from pathlib import Path

from nexa_sentimotion_filename_parser.metadata import Metadata
from nexa_py_sentimotion_mapper.sentimotion_mapper import Mapper
from nexa_preprocessing.cleaning.openface_data_cleaning import OpenfaceDataCleaner

from constants import AU_INTENSITY_COLS

Mapper._load_data_if_needed()

folder_path = "../data/openface_files"

furhat_folder = Path(folder_path) / "furhat"
metahuman_folder = Path(folder_path) / "metahuman"
original_folder = Path(folder_path) / "original"

folders = {"furhat": furhat_folder, "metahuman": metahuman_folder, "original": original_folder}

data = []

for condition, folder in folders.items():
    for filename in folder.glob("*.csv"):
        metadata = Metadata(filename.stem)
        row = {
            "condition": condition,
            "filename": filename.stem,
            "video_id": metadata.video_id,
            "emotion": Mapper.get_emotion_from_id(metadata.emotion_1_id),
            "intensity_level": metadata.intensity_level,
            "mode": metadata.mode,
        }

        temp_df = pd.read_csv(filename)

        # Filter for successful frames
        success_frames = temp_df[temp_df["success"] == 1]

        # Calculate mean and variance for action unit columns
        au_columns_mean = success_frames[AU_INTENSITY_COLS].mean()
        au_columns_var = success_frames[AU_INTENSITY_COLS].var()

        # Add mean values to the row dictionary with '_mean' suffix
        row.update({f"{col}_mean": mean_val for col, mean_val in au_columns_mean.items()})

        # Add variance values to the row dictionary with '_var' suffix
        row.update({f"{col}_var": var_val for col, var_val in au_columns_var.items()})

        data.append(row)

# Create DataFrame from the aggregated data
df = pd.DataFrame(data)
print(df["emotion"].unique())

# Save the DataFrame to a CSV file
df.to_csv("../data/out/openface_data.csv", index=False)
