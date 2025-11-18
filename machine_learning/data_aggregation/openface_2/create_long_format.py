import os
import pandas as pd
from constants import ROOT_DIR

# Path
openface_path = os.path.join(ROOT_DIR, "data/out/openface_data.csv")
df = pd.read_csv(openface_path)

# Keep only AU mean features
feature_cols = [c for c in df.columns if "mean" in c]

# Melt to long format
df_long = df.melt(
    id_vars=["condition", "video_id", "emotion"],   # metadata to keep
    value_vars=feature_cols,                             # AU features
    var_name="feature",
    value_name="activation"
)

# Save to CSV
out_path = os.path.join(ROOT_DIR, "data/out/openface_data_long_all.csv")
df_long.to_csv(out_path, index=False)

print(f"Saved long-format AU dataframe to: {out_path}")
print(df_long.head())