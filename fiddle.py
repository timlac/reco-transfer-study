import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from nexa_sentimotion_filename_parser.metadata import Metadata

from constants import ROOT_DIR
from box_scripts.test_dl import get_box_video_ids

df_openface_external = pd.read_csv(os.path.join(ROOT_DIR, "data/out/openface_data_external.csv"))
df_openface_internal = pd.read_csv(os.path.join(ROOT_DIR, "data/out/openface_data.csv"))

print(df_openface_external["emotion"].unique())
print(df_openface_external["emotion_id"].unique())



external_video_ids = set(df_openface_external["video_id"].dropna())
internal_video_ids = set(df_openface_internal["video_id"].dropna())
overlapping_video_ids = sorted(external_video_ids & internal_video_ids)
external_only_video_ids = sorted(external_video_ids - internal_video_ids)
internal_only_video_ids = sorted(internal_video_ids - external_video_ids)

summary = pd.DataFrame([
    {
        "Dataset": name,
        "Rows": len(df),
        "Unique video IDs": len(video_ids),
        "Overlapping IDs": len(overlapping_video_ids),
        "Non-overlapping IDs": len(only_video_ids),
        "Overlap (%)": round(100 * len(overlapping_video_ids) / len(video_ids), 2)
        if video_ids else float("nan"),
    }
    for name, df, video_ids, only_video_ids in [
        ("External", df_openface_external, external_video_ids, external_only_video_ids),
        ("Internal", df_openface_internal, internal_video_ids, internal_only_video_ids),
    ]
])

print(summary.to_string(index=False))
print(f"\nUnique video IDs across both datasets: {len(external_video_ids | internal_video_ids)}")
print(f"Non-overlapping video IDs across both datasets: {len(external_video_ids ^ internal_video_ids)}")

external_rows_without_overlap = (
    ~df_openface_external["video_id"].isin(overlapping_video_ids)
).sum()
print(f"External rows remaining after excluding overlapping video IDs: {external_rows_without_overlap}")

for label, video_ids in [
    ("Overlapping", overlapping_video_ids),
    ("External only", external_only_video_ids),
    ("Internal only", internal_only_video_ids),
]:
    print(f"\n{label} video IDs ({len(video_ids)}):")
    print(video_ids)


edited_video_ids, validation_video_ids = get_box_video_ids()
box_video_ids = edited_video_ids | validation_video_ids
new_video_ids = box_video_ids - internal_video_ids - external_video_ids

box_summary = pd.DataFrame([
    {
        "Box source": name,
        "Unique IDs": len(video_ids),
        "Internal overlap": len(video_ids & internal_video_ids),
        "Existing external only": len(video_ids & (external_video_ids - internal_video_ids)),
        "New eligible IDs": len(video_ids - internal_video_ids - external_video_ids),
    }
    for name, video_ids in [
        ("Edited", edited_video_ids),
        ("Validation", validation_video_ids),
        ("Combined (unique)", box_video_ids),
    ]
])

print("\nBox availability compared with the local datasets:")
print(box_summary.to_string(index=False))
print(f"\nIDs shared by Edited and Validation: {len(edited_video_ids & validation_video_ids)}")
print(f"Additional video IDs absent from both local datasets: {len(new_video_ids)}")
print(
    "Potential external unique video IDs after excluding internal overlap and adding new IDs: "
    f"{len(external_only_video_ids)} + {len(new_video_ids)} = "
    f"{len(external_only_video_ids) + len(new_video_ids)}"
)

download_candidates = pd.DataFrame([
    {
        "video_id": video_id,
        "Edited": video_id in edited_video_ids,
        "Validation": video_id in validation_video_ids,
    }
    for video_id in sorted(new_video_ids)
], columns=["video_id", "Edited", "Validation"])

print("\nNew download candidates and their Box sources:")
print(download_candidates.to_string(index=False))
print("Candidate counts are video IDs; additional usable rows require downloading and processing the clips.")
