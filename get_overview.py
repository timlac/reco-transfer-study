from fileinput import filename
from pathlib import Path
import pandas as pd

from nexa_sentimotion_filename_parser.metadata import Metadata

folder_path ="data"

furhat_folder = Path(folder_path) / "furhat"
metahuman_folder = Path(folder_path) / "metahuman"
original_folder = Path(folder_path) / "original"

furhat_count = len(list(furhat_folder.glob("*.mov")))
metahuman_count = len(list(metahuman_folder.glob("*.mov")))
original_count = len(list(original_folder.glob("*.mov")))

print(f'Furhat folder: {furhat_count} files')
print(f'Metahuman folder: {metahuman_count} files')
print(f'Original folder: {original_count} files')

metas = []

folders = {"furhat": furhat_folder, "metahuman": metahuman_folder, "original": original_folder}

for condition, folder in folders.items():
    for p in Path(folder).glob("*.mov"):
        print(p.stem)
        m = Metadata(p.stem)

        print(vars(m))
        print()

        rec = {
            "condition": condition,
            "filename": m.filename,
            "video_id": m.video_id,
            "emotion_1_abr": m.emotion_1_abr,
            "mode": m.mode,
            "intensity_level": m.intensity_level,
            "version": m.version,
            "situation": m.situation,
        }
        metas.append(rec)

df = pd.DataFrame(metas)

print(df[df["condition"] == "metahuman"]["emotion_1_abr"].value_counts())
print(df[df["condition"] == "furhat"]["emotion_1_abr"].value_counts())
print(df[df["condition"] == "original"]["emotion_1_abr"].value_counts())

print(df[df["condition"] == "metahuman"]["mode"].value_counts())
print(df[df["condition"] == "furhat"]["mode"].value_counts())
print(df[df["condition"] == "original"]["mode"].value_counts())