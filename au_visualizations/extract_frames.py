import os

videos_base = "/home/tim/Work/recordix/reco-transfer-study/data/visualization_example/A435_conf_v_3"

t_ref = 2.18

videos = {
    "original": f"{videos_base}/videos/original_A435_conf_v_3.mp4",
    "furhat": f"{videos_base}/videos/furhat_A435_conf_v_3.mp4",
    "metahuman": f"{videos_base}/videos/metahuman_A435_conf_v_3.mp4"
}

for name, vid in videos.items():
    out_path = f"{videos_base}/frames/{name}_frame.png"
    cmd = f"ffmpeg -y -ss {t_ref} -i {vid} -frames:v 1 {out_path}"
    print(cmd)
    os.system(cmd)

print("All frames extracted!")
