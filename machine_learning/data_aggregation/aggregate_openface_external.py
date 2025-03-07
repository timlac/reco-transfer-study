import pandas as pd
from pathlib import Path
import os
from nexa_sentimotion_filename_parser.metadata import Metadata
from nexa_py_sentimotion_mapper.sentimotion_mapper import Mapper
from nexa_preprocessing.cleaning.openface_data_cleaning import OpenfaceDataCleaner

from constants import openface_feature_columns, ROOT_DIR
from machine_learning.data_aggregation.utils import update_row

Mapper._load_data_if_needed()

data_folder = Path("/media/tim/Seagate Hub/transfer_study_openface_files/")

data = []


for filename in data_folder.glob("*.csv"):
    print(f"\nProcessing {filename}")

    metadata = Metadata(filename.stem)
    row = {
        "filename": filename.stem,
        "video_id": metadata.video_id,
        "emotion": Mapper.get_emotion_from_id(metadata.emotion_1_id),
        "emotion_id": metadata.emotion_1_id,
        "intensity_level": metadata.intensity_level,
        "mode": metadata.mode,
    }

    temp_df = pd.read_csv(filename)
    print(f"Number of processed frames: {len(temp_df)}")
    if temp_df.empty:
        print(f"Empty DataFrame for {filename}")
        continue

    # Filter for successful frames
    success_frames = temp_df[temp_df["success"] == 1]
    print(f"Number of successful frames: {len(success_frames)}")

    if success_frames.empty:
        print(f"No successful frames for {filename}")
        continue

    # Calculate mean and variance for action unit columns
    means = success_frames[openface_feature_columns].mean()
    varss = success_frames[openface_feature_columns].var()
    quantile_20 = success_frames[openface_feature_columns].quantile(0.2)
    quantile_50 = success_frames[openface_feature_columns].quantile(0.5)
    quantile_80 = success_frames[openface_feature_columns].quantile(0.8)
    iqr_values = quantile_80 - quantile_20

    update_row(row, means, "_mean")
    update_row(row, varss, "_var")
    update_row(row, quantile_20, "_20th")
    update_row(row, quantile_50, "_50th")
    update_row(row, quantile_80, "_80th")
    update_row(row, iqr_values, "_iqr")

    data.append(row)

# Create DataFrame from the aggregated data
df = pd.DataFrame(data)
print(df["emotion"].unique())

# Save the DataFrame to a CSV file
df.to_csv(os.path.join(ROOT_DIR, "data/out/openface_data_external.csv"), index=False)
