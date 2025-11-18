from pathlib import Path
import pandas as pd
import os

from nexa_py_sentimotion_mapper.sentimotion_mapper import Mapper
from nexa_sentimotion_filename_parser.metadata import Metadata

from constants import ROOT_DIR

data_path = Path(os.path.join(ROOT_DIR, "data/opensmile_files"))

df_list = []  # List to store individual DataFrames

for fp in data_path.glob('*.csv'):
    metadata = Metadata(fp.stem)

    # Read the single-row DataFrame
    temp_df = pd.read_csv(fp)

    temp_df.drop(columns=["file", "start", "end"], inplace=True)
    
    print(temp_df.columns)

    # Insert metadata columns at the beginning
    temp_df.insert(0, "filename", fp.stem)
    temp_df.insert(1, "video_id", metadata.video_id)
    temp_df.insert(2, "emotion", Mapper.get_emotion_from_id(metadata.emotion_1_id))
    temp_df.insert(3, "emotion_id", metadata.emotion_1_id)
    temp_df.insert(4, "intensity_level", metadata.intensity_level)
    temp_df.insert(5, "mode", metadata.mode)

    df_list.append(temp_df)

# Concatenate all DataFrames
final_df = pd.concat(df_list, ignore_index=True)

final_df.to_csv(os.path.join(ROOT_DIR, "data/out/opensmile_data.csv"), index=False)


