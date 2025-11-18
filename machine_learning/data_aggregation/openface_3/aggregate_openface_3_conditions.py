import pandas as pd
from pathlib import Path
import os
from nexa_sentimotion_filename_parser.metadata import Metadata
from nexa_py_sentimotion_mapper.sentimotion_mapper import Mapper
from nexa_preprocessing.cleaning.openface_data_cleaning import OpenfaceDataCleaner
import ast
from constants import openface_feature_columns, ROOT_DIR, OPENFACE_3_AUS, openface_3_feature_columns
from machine_learning.data_aggregation.utils import update_row

Mapper._load_data_if_needed()

folder_path = os.path.join(ROOT_DIR, "data/openface_3_files")

furhat_folder = Path(folder_path) / "furhat"
metahuman_folder = Path(folder_path) / "metahuman"
original_folder = Path(folder_path) / "original"

folders = {"furhat": furhat_folder, "metahuman": metahuman_folder, "original": original_folder}

data = []

for condition, folder in folders.items():
    for filename in folder.glob("*.csv"):

        filename_no_prepend = filename.stem.replace("face_analysis_", "")

        metadata = Metadata(filename_no_prepend)
        row = {
            "condition": condition,
            "filename": filename_no_prepend,
            "video_id": metadata.video_id,
            "emotion": Mapper.get_emotion_from_id(metadata.emotion_1_id),
            "emotion_id": metadata.emotion_1_id,
            "intensity_level": metadata.intensity_level,
            "mode": metadata.mode,
        }

        temp_df = pd.read_csv(filename)

        # Filter for successful frames
        success_frames = temp_df[temp_df["confidence"] > 0.90]

        # 1) Turn string "[[...]]" into a flat Python list [v1, ..., v8]
        success_frames["action_units_parsed"] = success_frames["action_units"].apply(
            lambda x: ast.literal_eval(x)[0]  # from [[...]] -> [...]
        )

        # 2) Expand that list into 8 separate AU columns
        success_frames[OPENFACE_3_AUS] = pd.DataFrame(
            success_frames["action_units_parsed"].tolist(),
            index=success_frames.index,
        )

        # Calculate mean and variance for action unit columns
        means = success_frames[openface_3_feature_columns].mean()
        varss = success_frames[openface_3_feature_columns].var()
        quantile_20 = success_frames[openface_3_feature_columns].quantile(0.2)
        quantile_50 = success_frames[openface_3_feature_columns].quantile(0.5)
        quantile_80 = success_frames[openface_3_feature_columns].quantile(0.8)
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
df.to_csv(os.path.join(ROOT_DIR, "data/out/openface_3_data.csv"), index=False)
