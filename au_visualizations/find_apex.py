import pandas as pd
import numpy as np

def select_best_frame(csv_path):
    df = pd.read_csv(csv_path)



    # keep only good tracking
    df = df[(df['success'] == 1) & (df['confidence'] > 0.9)]

    # print(df)

    # # near-frontal (tweak thresholds if needed; values in radians)
    # df = df[df['pose_Ry'].abs() < 0.2]   # yaw
    # df = df[df['pose_Rx'].abs() < 0.2]   # pitch

    # print(df)

    # pick AU intensity columns
    au_cols = [c for c in df.columns if c.endswith('_r')]

    # print(au_cols)
    # print("hej")
    df_au = df[au_cols]

    # print(df_au.describe())
    # "AU energy" as a proxy for apex
    df['au_energy'] = np.sqrt((df[au_cols] ** 2).sum(axis=1))

    # print(df["au_energy"])

    # print(df["au_energy"].describe())

    best = df.sort_values('au_energy', ascending=False).iloc[0]

    # print(best)

    return int(best['frame']), float(best['timestamp'])


# ---- UPDATED PATHS ----
base = "/home/tim/Work/recordix/reco-transfer-study/data/visualization_example/A435_conf_v_3/openface_2"

for name in [
    "furhat_A435_conf_v_3",
    "original_A435_conf_v_3",
    "metahuman_A435_conf_v_3",
]:
    csv_path = f"{base}/{name}.csv"

    print("Processing:", csv_path)
    frame, ts = select_best_frame(csv_path)
    print(name, "→ frame:", frame, "timestamp:", ts)
