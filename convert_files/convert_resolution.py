import subprocess
import os
from pathlib import Path

folder_path = "../data/videos"

furhat_folder = Path(folder_path) / "furhat"
metahuman_folder = Path(folder_path) / "metahuman"
original_folder = Path(folder_path) / "original"

folders = {"furhat": furhat_folder, "metahuman": metahuman_folder, "original": original_folder}

# Create a new folder for each condition
output_suffix = "_converted"

for condition, folder in folders.items():
    print(f"Processing condition: {condition}")

    # Ensure input folder exists
    if not folder.exists():
        print(f"Folder not found: {folder}")
        continue

    # Create output folder
    output_folder = folder.parent / (folder.name + output_suffix)
    output_folder.mkdir(exist_ok=True)

    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() == ".mov":
            output_file = output_folder / (file.stem + ".mp4")
            print(f"Converting {file.name} to {output_file.name} with resolution 1280x720 and audio bitrate 192k")

            try:
                # FFmpeg command to convert to MP4, set resolution, and audio bitrate
                subprocess.run([
                    "ffmpeg",
                    "-i", str(file),
                    "-vf", "scale=1280:720",
                    "-c:v", "libx264",
                    "-crf", "23",
                    "-preset", "medium",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    str(output_file)
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error converting {file.name}: {e}")

print("Processing complete.")
