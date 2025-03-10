import pandas as pd
import os
from glob import glob

from constants import ROOT_DIR

path = os.path.join(ROOT_DIR, "data/export")

export_glob = glob(path + "/*.csv")

dataframes = []

for file in export_glob:
    df = pd.read_csv(file)
    dataframes.append(df)

df = pd.concat(dataframes, ignore_index=True)

df["generic_id"] = df["user_id"].str.extract(r"(\d+)$").astype(int)
df["condition"] = df["user_id"].str.extract(r"^(.*?)(?=\d*$)")
df["accurate"] = df["emotion_1_id"] == df["reply"]


df.to_csv(os.path.join(ROOT_DIR, "data/export/merged_surveys.csv"), index=False)