import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from constants import ROOT_DIR

df_openface = pd.read_csv(os.path.join(ROOT_DIR, "data/out/openface_data.csv"))

emotions = df_openface["emotion"].unique()
print(emotions)