import subprocess
import os
from pathlib import Path

folder_path = "../data/videos"

furhat_folder = Path(folder_path) / "furhat"
metahuman_folder = Path(folder_path) / "metahuman"
original_folder = Path(folder_path) / "original"

folders = {"furhat": furhat_folder, "metahuman": metahuman_folder, "original": original_folder}

for condition, folder in folders.items():
    print(f"Processing condition: {condition}")

    # Ensure input folder exists
    if not folder.exists():
        print(f"Folder not found: {folder}")
        continue

    # Define output directories
    video_output_dir = folder / "video"
    audio_output_dir = folder / "audio"

    # Create output directories if needed
    video_output_dir.mkdir(parents=True, exist_ok=True)
    if condition == "original":
        audio_output_dir.mkdir(parents=True, exist_ok=True)

    # Loop through each file in the input directory
    for filename in os.listdir(folder):
        if filename.endswith('.mov'):
            input_path = folder / filename
            video_output_path = video_output_dir / filename

            if condition == "original":
                # Create audio output path for original condition
                audio_output_path = audio_output_dir / f"{filename.replace('.mov', '.aac')}"

                # Extract audio using FFmpeg (transcoding for compatibility)
                subprocess.run(
                    ['ffmpeg', '-i', str(input_path), '-vn', '-acodec', 'aac', '-b:a', '192k', str(audio_output_path)],
                    check=True
                )
                print(f"Audio extracted: {audio_output_path}")

            # Extract video (no audio) using FFmpeg
            subprocess.run(['ffmpeg', '-i', str(input_path), '-an', '-vcodec', 'copy', str(video_output_path)],
                           check=True)
            print(f"Video extracted: {video_output_path}")

print("Processing completed.")
