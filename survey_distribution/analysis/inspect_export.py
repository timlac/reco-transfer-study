import pandas as pd
import os
from glob import glob

path = "../../data/export"

export_glob = glob(path + "/*.csv")

dataframes = []

for file in export_glob:
    df = pd.read_csv(file)
    dataframes.append(df)

df = pd.concat(dataframes, ignore_index=True)

df["generic_id"] = df["user_id"].str.extract(r"(\d+)$").astype(int)


df.to_csv("../data/export/merged_surveys.csv", index=False)